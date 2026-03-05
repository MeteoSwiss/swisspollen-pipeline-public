#include <map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <zip.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "single_include/nlohmann/json.hpp"
#include <atomic>
#include <cuda_runtime.h>

using json = nlohmann::json;
using namespace std::chrono;

// ---------- Simple static thread pool ----------
class ThreadPool {
public:
    explicit ThreadPool(size_t n) : stop(false) {
        for (size_t i = 0; i < n; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> job;
                    {
                        std::unique_lock<std::mutex> lock(mtx);
                        cv.wait(lock, [this] { return stop || !jobs.empty(); });
                        if (stop && jobs.empty()) return;
                        job = std::move(jobs.front());
                        jobs.pop();
                    }
                    job();
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stop = true;
        }
        cv.notify_all();
        for (auto& t : workers) t.join();
    }

    void enqueue(std::function<void()> job) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            jobs.push(std::move(job));
        }
        cv.notify_one();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> jobs;
    std::mutex mtx;
    std::condition_variable cv;
    bool stop;
};

// ---------- Structures ----------
struct EventFiles {
    std::string img0, img1, json_file;
    bool is_complete() const { return !img0.empty() && !img1.empty() && !json_file.empty(); }
};

struct LoadedEvent {
    std::vector<uchar> img0, img1, json;
};

// ---------- ZIP loader ----------
std::vector<uchar> load_from_zip(zip_t* archive, const std::string& filename) {
    zip_stat_t st{};
    if (zip_stat(archive, filename.c_str(), 0, &st) != 0)
        throw std::runtime_error("File not found: " + filename);

    std::vector<uchar> buffer(st.size);
    zip_file_t* f = zip_fopen(archive, filename.c_str(), 0);
    if (!f) throw std::runtime_error("zip_fopen failed");

    zip_fread(f, buffer.data(), st.size);
    zip_fclose(f);
    return buffer;
}

// ----------- QUE Class -------------------
template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : capacity(capacity) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_not_full.wait(lock, [&] { return queue.size() < capacity; });
        queue.push(std::move(item));
        cv_not_empty.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx);
        cv_not_empty.wait(lock, [&] { return !queue.empty() || finished; });

        if (queue.empty())
            return false;

        item = std::move(queue.front());
        queue.pop();
        cv_not_full.notify_one();
        return true;
    }

    void set_finished() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            finished = true;
        }
        cv_not_empty.notify_all();
    }

private:
    std::queue<T> queue;
    size_t capacity;
    std::mutex mtx;
    std::condition_variable cv_not_full, cv_not_empty;
    bool finished = false;
};

// ---------- MAIN ----------
int main(int argc, char* argv[]) {

    std::string model_path = "model/meteoswiss_2025_Q2_15sp.onnx";
    std::string input_path;

    bool verbose = false;
    bool unittest = false;
    int num_threads = 12;
    int batch_size = 12;

    for (int i = 1; i < argc; i++) {

        std::string arg = argv[i];

         if (arg == "-u" || arg == "--unittest") {
            // model unitest configuration
            unittest = true;
            model_path = "model/meteoswiss_2025_Q2_15sp.onnx";
            input_path = "model_unitest/unitestdata.zip";
            verbose = true;
        }
        else if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            model_path = argv[++i];
        }
        else if ((arg == "--input" || arg == "-i") && i + 1 < argc) {
            input_path = argv[++i];
        }
        else if (arg == "--num_threads" && i + 1 < argc) {
            num_threads = std::stoi(argv[++i]);
        }
        else if (arg == "--batch_size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        }
        else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        }
        else if (arg == "--help" || arg == "-h") {

            std::cout << "Usage:\n";
            std::cout << argv[0] << " -m model.onnx -i data.zip [options]\n\n";

            std::cout << "Options:\n";
            std::cout << "  -m, --model PATH        ONNX model file\n";
            std::cout << "  -i, --input PATH        Input zip file\n";
            std::cout << "  --num_threads N         Number of threads (default: 12)\n";
            std::cout << "  --batch_size N          Batch size (default: 12)\n";
            std::cout << "  -v, --verbose           Print class and score\n";
            std::cout << "  -h, --help              Show this help\n";

            return 0;
        }
    }

    if (model_path.empty() || input_path.empty()) {
        std::cerr << "Error: model and input required\n";
        std::cerr << "Use -h for help\n";
        return 1;
    }

    std::cout << "------------------- run params -----------------------" << std::endl;
    if (unittest) std::cout << "Running UNIT TEST configuration\n";
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Input: " << input_path << std::endl;
    std::cout << "Threads: " << num_threads << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;   
    if (verbose) std::cout << "Verbose mode enabled\n";
    std::cout << "------------------------------------------------------" << std::endl<< std::endl;
    time_point<high_resolution_clock> t3;
    auto t1 = high_resolution_clock::now();
    
    try {
 
        // ================= ONNX INIT (TENSORRT V2) =================
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "parallel_pool");
        const auto& api = Ort::GetApi();

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(2);
        session_options.SetInterOpNumThreads(2);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Create the TensorRT V2 options structure
        OrtTensorRTProviderOptionsV2* trt_options = nullptr;
        api.CreateTensorRTProviderOptions(&trt_options);

        // Define TensorRT-specific keys
        std::vector<const char*> keys = {
            "device_id",
            "trt_max_workspace_size",
            "trt_fp16_enable",
            "trt_engine_cache_enable",
            "trt_engine_cache_path",
            // --- NEW TUNABLES ---
            "trt_builder_optimization_level"   
        };

        std::vector<const char*> values = {
            "0",
            "8589934592",      // 8GB Workspace (Give it everything you can spare during build)
            "0",               // FP16  Maybe safe but we should reexport or sanitize onnx ?????
            "1",               // Cache Enable
            "./trt_engines",   
            // --- NEW VALUES ---
            "4"              // Max Optimization (Build takes longer, inference is faster)            
        };

        // Apply options and append provider
        api.UpdateTensorRTProviderOptions(trt_options, keys.data(), values.data(), keys.size());
        session_options.AppendExecutionProvider_TensorRT_V2(*trt_options);

        // Load the model (TRT Compilation happens here on first run)
        // Note: First run will be SLOW (building engine). Subsequent runs will be fast (loading cache).
        Ort::Session session(env, model_path.c_str(), session_options);

        api.ReleaseTensorRTProviderOptions(trt_options);

        Ort::AllocatorWithDefaultOptions allocator;


     
        auto in0 = session.GetInputNameAllocated(0, allocator);
        auto in1 = session.GetInputNameAllocated(1, allocator);
        auto in2 = session.GetInputNameAllocated(2, allocator);
        const char* input_names[] = { in0.get(), in1.get(), in2.get() };

        auto out0 = session.GetOutputNameAllocated(0, allocator);
        const char* output_names[] = { out0.get() };

        // ===================== ATOMIC TIMERS =====================
        std::atomic<long long> total_decode_us{0};
        std::atomic<long long> total_tensor_us{0};
        std::atomic<long long> total_infer_us{0};
        std::atomic<long long> total_post_us{0};
        std::atomic<long long> total_event_us{0};
        std::atomic<size_t> total_events{0};

        auto t2 = high_resolution_clock::now();
        std::cout << duration_cast<milliseconds>(t2 - t1).count() << " ms onnx init\n";

        // ================= ZIP INIZIALIZE =================
        int err = 0;
        zip_t* archive = zip_open(input_path.c_str(), 0, &err);
        if (!archive) throw std::runtime_error("Cannot open zip");

        std::map<std::string, EventFiles> event_map;
        int n = zip_get_num_entries(archive, 0);

        for (int i = 0; i < n; i++) {
            std::string name = zip_get_name(archive, i, 0);

            if (name.find(".json") != std::string::npos)
                event_map[name.substr(0, name.find(".json"))].json_file = name;

            else if (name.find(".0.0.rec_mag.png") != std::string::npos)
                event_map[name.substr(0, name.find(".computed_data"))].img0 = name;

            else if (name.find(".0.1.rec_mag.png") != std::string::npos)
                event_map[name.substr(0, name.find(".computed_data"))].img1 = name;
        }

        BoundedQueue<LoadedEvent> queue(512);  // limit memory

        std::thread producer([&] {
            for (auto& [_, f] : event_map) {
                if (!f.is_complete()) continue;

                LoadedEvent ev;
                ev.img0 = load_from_zip(archive, f.img0);
                ev.img1 = load_from_zip(archive, f.img1);
                ev.json = load_from_zip(archive, f.json_file);

                queue.push(std::move(ev));
            }
            queue.set_finished();
        });      
        
        t3 = high_resolution_clock::now();
        std::cout << duration_cast<milliseconds>(t3 - t2).count() << " ms read zip index (queing zip fn's)\n";

        // ---------------- THREAD POOL ----------------
        {
            ThreadPool pool(num_threads);

            // enqueue persistent workers
            for (size_t t = 0; t < num_threads; ++t) {
                pool.enqueue([&] {
                    const int H = 200, W = 200;
                    cv::Mat tmp0(H, W, CV_32FC1), tmp1(H, W, CV_32FC1);
                    std::vector<float> img0_single(H*W), img1_single(H*W), json_data_single(13);

                    // Preallocate batch memory
                    std::vector<float> img0_batch(batch_size * H * W);
                    std::vector<float> img1_batch(batch_size * H * W);
                    std::vector<float> json_batch(batch_size * 13);

                    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

                    LoadedEvent e;
                    while (true) {
                        // --- Collect a batch ---
                        size_t b = 0;
                        
                        auto t_batchread_start = std::chrono::high_resolution_clock::now();

                        for (; b < batch_size && queue.pop(e); ++b) {
                            // ---------------- Decode & normalize ----------------
                            cv::Mat tmp0f, tmp1f;
                            // Decode
                            tmp0 = cv::imdecode(e.img0, cv::IMREAD_GRAYSCALE);
                            tmp1 = cv::imdecode(e.img1, cv::IMREAD_GRAYSCALE);
                            // Resize if needed
                            if(tmp0.size() != cv::Size(W,H)) cv::resize(tmp0, tmp0, {W,H});
                            if(tmp1.size() != cv::Size(W,H)) cv::resize(tmp1, tmp1, {W,H});
                            // Convert to float 32-bit in range [0,1]
                            tmp0.convertTo(tmp0f, CV_32FC1, 1.f/255.f);
                            tmp1.convertTo(tmp1f, CV_32FC1, 1.f/255.f);
                            // Normalize: (x - 0.485)/0.229
                            tmp0f -= 0.485f; tmp0f /= 0.229f;
                            tmp1f -= 0.485f; tmp1f /= 0.229f;
                            //tmp0f = (tmp0f - 0.485f) / 0.229f;
                            //tmp1f = (tmp1f - 0.485f) / 0.229f;
                            // Copy to single-event vectors
                            std::memcpy(img0_single.data(), tmp0f.ptr<float>(), H*W*sizeof(float));
                            std::memcpy(img1_single.data(), tmp1f.ptr<float>(), H*W*sizeof(float));

                            // ------------------ LOAD JSON ------------------
                            json event_data = json::parse(e.json.begin(), e.json.end());
                            std::vector<float> fl_spectra;
                            auto rel_spec_ptr = event_data["computed_data"]
                                .value("fluorescence", json::object())
                                .value("processed_data", json::object())
                                .value("spectra", json::object())
                                .value("relative_spectra", json());
                            if (!rel_spec_ptr.is_null() && rel_spec_ptr.is_array()) {
                                for (const auto& sublist : rel_spec_ptr) {
                                    if (sublist.is_array()) {
                                        for (const auto& val : sublist) fl_spectra.push_back(val.get<float>());
                                    }
                                }
                            }
                            std::fill(json_data_single.begin(), json_data_single.end(), 0.f);
                            for (size_t k = 0, j = 0; k < fl_spectra.size(); ++k) {
                                if (k == 5 || k == 10) continue;
                                json_data_single[j++] = fl_spectra[k];
                            }

                            // Copy into batch memory
                            std::copy(img0_single.begin(), img0_single.end(), img0_batch.begin() + b*H*W);
                            std::copy(img1_single.begin(), img1_single.end(), img1_batch.begin() + b*H*W);
                            std::copy(json_data_single.begin(), json_data_single.end(), json_batch.begin() + b*13);
                        }
                        auto t_batchread_end = std::chrono::high_resolution_clock::now();


                        if (b == 0) break; // queue is finished

                        // ---------------- Create batched tensors ----------------
                        std::array<int64_t,4> img_shape{(int64_t)b, 1, H, W};
                        std::array<int64_t,2> json_shape{(int64_t)b, 13};

                        std::array<Ort::Value,3> inputs = {
                            Ort::Value::CreateTensor<float>(mem, img0_batch.data(), b*H*W, img_shape.data(), img_shape.size()),
                            Ort::Value::CreateTensor<float>(mem, img1_batch.data(), b*H*W, img_shape.data(), img_shape.size()),
                            Ort::Value::CreateTensor<float>(mem, json_batch.data(), b*13, json_shape.data(), json_shape.size())
                        };

                        // ---------------- Run inference ----------------
                        auto t_infer_start = std::chrono::high_resolution_clock::now();
                        auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), inputs.size(),
                                                output_names, 1);
                        auto t_infer_end = std::chrono::high_resolution_clock::now();

                        // ---------------- Process outputs ----------------
                        float* scores = outputs[0].GetTensorMutableData<float>();
                        size_t ncls = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
                        size_t per_event = ncls / b;

                        for (size_t i = 0; i < b; ++i) {
                            int winning_class = std::distance(scores + i*per_event,
                                                            std::max_element(scores + i*per_event, scores + (i+1)*per_event));
                            if (verbose) {
                                std::cout << "  - Winning Class: " << winning_class << " (Score: " << scores[winning_class] << ")" << std::endl;
                            }
                            total_events++;
                        }
                        auto t_post_end = std::chrono::high_resolution_clock::now();

                        // ---------------- Aggregate timing ----------------
                        total_decode_us += std::chrono::duration_cast<std::chrono::microseconds>(t_batchread_end - t_batchread_start).count()/b;
                        total_tensor_us += std::chrono::duration_cast<std::chrono::microseconds>(t_infer_start - t_batchread_end).count()/b;
                        total_infer_us += std::chrono::duration_cast<std::chrono::microseconds>(t_infer_end - t_infer_start).count()/b;
                        total_post_us += std::chrono::duration_cast<std::chrono::microseconds>(t_post_end - t_infer_end).count()/b;
                        total_event_us += std::chrono::duration_cast<std::chrono::microseconds>(t_post_end - t_batchread_start).count()/b;
                    }
                });

            }

            producer.join();
            zip_close(archive);

            std::cout << "Processed events: " << total_events.load() << "\n";
            std::cout << "Average timings per event (ms), independent of parallelization:\n"
                    << "  Decode png & json  : " << total_decode_us.load() / 1000.0 / total_events.load() << "\n"
                    << "  build Tensor       : " << total_tensor_us.load() / 1000.0 / total_events.load() << "\n"
                    << "  Inference          : " << total_infer_us.load() / 1000.0 / total_events.load() << "\n"
                    << "  Postprocessing     : " << total_post_us.load() / 1000.0 / total_events.load() << "\n"
                    << "  Total              : " << total_event_us.load() / 1000.0 / total_events.load() << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
   
    auto t5 = high_resolution_clock::now();
    std::cout << duration_cast<milliseconds>(t5 - t3).count() << " ms decode and Infere block\n";
    std::cout << duration_cast<milliseconds>(t5 - t1).count() << " ms TOTAL\n";
}

#include <kompute/Kompute.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

static std::vector<uint32_t> compileSource(const std::string& source)
{
    std::ofstream fileOut("tmp_kp_shader.comp");
    fileOut << source;
    fileOut.close();
    if (system(
          std::string(
            "glslangValidator -V tmp_kp_shader.comp -o tmp_kp_shader.comp.spv")
            .c_str()))
        throw std::runtime_error("Error running glslangValidator command");
    std::ifstream fileStream("tmp_kp_shader.comp.spv", std::ios::binary);
    std::vector<char> buffer;
    buffer.insert(
      buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
    return { (uint32_t*)buffer.data(),
             (uint32_t*)(buffer.data() + buffer.size()) };
}

const std::vector<unsigned char> s = {
    #include "simpletest.comp.spv.h"
};

void kompute() {

    // 1. Create Kompute Manager with default settings (device 0, first queue and no extensions)
    kp::Manager mgr; 

    // 2. Create and initialise Kompute Tensors through manager

    // Default tensor constructor simplifies creation of float values
    auto tensorInA = mgr.tensor({ 2., 2., 2. });
    auto tensorInB = mgr.tensor({ 1., 2., 3. });
    // Explicit type constructor supports uint32, int32, double, float and bool
    auto tensorOutA = mgr.tensorT<uint32_t>({ 0, 0, 0 });
    auto tensorOutB = mgr.tensorT<uint32_t>({ 0, 0, 0 });

    std::vector<std::shared_ptr<kp::Tensor>> params = {tensorInA, tensorInB, tensorOutA, tensorOutB};

    // 3. Create algorithm based on shader (supports buffers & push/spec constants)
    kp::Workgroup workgroup({3, 1, 1});
    std::vector<float> specConsts({ 2 });
    std::vector<float> pushConstsA({ 2.0 });
    std::vector<float> pushConstsB({ 3.0 });

    auto algorithm = mgr.algorithm(params,
                                   // See documentation shader section for compileSource
                                   { (uint32_t*)s.data(), (uint32_t*)(s.data() + s.size()) },
                                   workgroup,
                                   specConsts,
                                   pushConstsA);

    // 4. Run operation synchronously using sequence
    mgr.sequence()
        ->record<kp::OpTensorSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algorithm) // Binds default push consts
        ->eval() // Evaluates the two recorded operations
        ->record<kp::OpAlgoDispatch>(algorithm, pushConstsB) // Overrides push consts
        ->eval(); // Evaluates only last recorded operation

    // 5. Sync results from the GPU asynchronously
    auto sq = mgr.sequence();
    sq->evalAsync<kp::OpTensorSyncLocal>(params);

    // ... Do other work asynchronously whilst GPU finishes

    sq->evalAwait();

    // Prints the first output which is: { 4, 8, 12 }
    for (const float& elem : tensorOutA->vector()) std::cout << elem << "  ";
    // Prints the second output which is: { 10, 10, 10 }
    for (const float& elem : tensorOutB->vector()) std::cout << elem << "  ";

} // Manages / releases all CPU and GPU memory resources

int main()
{
    kompute();
}
#include <jni.h>
#include <string>
#include <chrono>
#include <arm_neon.h>


extern "C" JNIEXPORT jstring JNICALL
Java_com_shinplest_neonintrinsic_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

short *generateRamp(short startValue, short len) {
    short *ramp = new short[len];
    for (short i = 0; i < len; i++) {
        ramp[i] = startValue + i;
    }
    return ramp;
}

double msElapsedTime(std::chrono::system_clock::time_point start) {
    auto end = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

std::chrono::system_clock::time_point now() {
    return std::chrono::system_clock::now();
}

int dotProduct(short *vector1, short *vector2, short len) {
    int result = 0;
    for (short i = 0; i < len; i++) { result += vector1[i] * vector2[i]; }
    return result;
}

int dotProductNeon(short *vector1, short *vector2, short len) {
    const short transferSize = 4;
    short segments = len / transferSize;
    int32x4_t partialSumsNeon = vdupq_n_s32(0);

    for (short i = 0; i < segments; i++) {
        short offset = i * transferSize;
        int16x4_t vector1Neon = vld1_s16(vector1 + offset);
        int16x4_t vector2Neon = vld1_s16(vector2 + offset);
        partialSumsNeon = vmlal_s16(partialSumsNeon, vector1Neon, vector2Neon);
    } // Store partial sums int partial
    int partialSums[transferSize];
    vst1q_s32(partialSums, partialSumsNeon);// Sum up partial sums
    int result = 0;
    for (short i = 0; i < transferSize; i++) { result += partialSums[i]; }
    return result;
}
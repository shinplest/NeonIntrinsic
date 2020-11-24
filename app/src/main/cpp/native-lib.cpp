#include <jni.h>
#include <string>
#include <chrono>
#include <arm_neon.h>


short *generateRamp(short startValue, short len) {
    short *ramp = new short[len];
    for (short i = 0; i < len; i++) {
        ramp[i] = startValue + i;
    }
    return ramp;
}


int dotProduct(short *vector1, short *vector2, short len) {
    int result = 0;

    for (short i = 0; i < len; i++) {
        result += vector1[i] * vector2[i];
    }
    return result;
}

double msElapsedTime(std::chrono::system_clock::time_point start) {
    auto end = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
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
    for (short i = 0; i < transferSize; i++) {
        result += partialSums[i];
    }
    return result;
}

std::chrono::system_clock::time_point now() {
    return std::chrono::system_clock::now();
}


int dotProductNeonUroll(short *vector1, short *vector2, short len) {
    const short transferSize = 4;
    short segments = len / transferSize;

    // 4-element vector of zeros
    int32x4_t partialSumsNeon = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);
    int32x4_t sum2 = vdupq_n_s32(0);
    int32x4_t sum3 = vdupq_n_s32(0);
    int32x4_t sum4 = vdupq_n_s32(0);

    // Main loop (note that loop index goes through segments). Unroll with 4
    int i = 0;
    for (; i + 3 < segments; i += 4) {
        // Preload may help speed up sometimes
        // asm volatile("prfm pldl1keep, [%0, #256]" : :"r"(vector1) :);
        // asm volatile("prfm pldl1keep, [%0, #256]" : :"r"(vector2) :);

        // Load vector elements to registers
        int16x8_t v11 = vld1q_s16(vector1);
        int16x4_t v11_low = vget_low_s16(v11);
        int16x4_t v11_high = vget_high_s16(v11);

        int16x8_t v12 = vld1q_s16(vector2);
        int16x4_t v12_low = vget_low_s16(v12);
        int16x4_t v12_high = vget_high_s16(v12);

        int16x8_t v21 = vld1q_s16(vector1 + 8);
        int16x4_t v21_low = vget_low_s16(v21);
        int16x4_t v21_high = vget_high_s16(v21);

        int16x8_t v22 = vld1q_s16(vector2 + 8);
        int16x4_t v22_low = vget_low_s16(v22);
        int16x4_t v22_high = vget_high_s16(v22);

        // Multiply and accumulate: partialSumsNeon += vector1Neon * vector2Neon
        sum1 = vmlal_s16(sum1, v11_low, v12_low);
        sum2 = vmlal_s16(sum2, v11_high, v12_high);
        sum3 = vmlal_s16(sum3, v21_low, v22_low);
        sum4 = vmlal_s16(sum4, v21_high, v22_high);

        vector1 += 16;
        vector2 += 16;
    }
    partialSumsNeon = sum1 + sum2 + sum3 + sum4;

    // Sum up remain parts
    int remain = len % transferSize;
    for (i = 0; i < remain; i++) {

        int16x4_t vector1Neon = vld1_s16(vector1);
        int16x4_t vector2Neon = vld1_s16(vector2);
        partialSumsNeon = vmlal_s16(partialSumsNeon, vector1Neon, vector2Neon);

        vector1 += 4;
        vector2 += 4;
    }

    // Store partial sums
    int partialSums[transferSize];
    vst1q_s32(partialSums, partialSumsNeon);

    // Sum up partial sums
    int result = 0;
    for (int i = 0; i < transferSize; i++) {
        result += partialSums[i];
    }

    return result;
}


extern "C" JNIEXPORT jstring JNICALL
Java_com_shinplest_neonintrinsic_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    const int rampLength = 1024 * 8;
    const int trials = 10000;

    auto ramp1 = generateRamp(0, rampLength);
    auto ramp2 = generateRamp(100, rampLength);

    int lastResult = 0;
    auto start = now();
    for (int i = 0; i < trials; i++) {
        lastResult = dotProduct(ramp1, ramp2, rampLength);
    }
    auto elapsedTime = msElapsedTime(start);

    int lastResultNeon = 0;
    start = now();
    for (int i = 0; i < trials; i++) {
        lastResultNeon = dotProductNeon(ramp1, ramp2, rampLength);
    }
    auto elapsedTimeNeon = msElapsedTime(start);

    int lastResultNeonUnroll = 0;
    start = now();
    for (int i = 0; i < trials; i++) {
        lastResultNeonUnroll = dotProductNeonUroll(ramp1, ramp2, rampLength);
    }
    auto elapsedTimeNeonUnroll = msElapsedTime(start);

    delete ramp1, ramp2;
    std::string resultsString =
            "----==== NO NEON ====----\nResult: " + std::to_string(lastResult)
            + "\nElapsed time: " + std::to_string((int) elapsedTime) + " ms"
            + "\n\n----==== NEON ====----\n"
            + "Result: " + std::to_string(lastResultNeon)
            + "\nElapsed time: " + std::to_string((int) elapsedTimeNeon) + " ms"
            + "\n\n----==== NEON Unroll====----\n"
            + "Result: " + std::to_string(lastResultNeonUnroll)
            + "\nElapsed time: " + std::to_string((int) elapsedTimeNeonUnroll) + " ms";

    return env->NewStringUTF(resultsString.c_str());
}


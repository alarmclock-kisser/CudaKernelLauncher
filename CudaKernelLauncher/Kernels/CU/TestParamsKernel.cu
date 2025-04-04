extern "C" __global__ void TestParams(float* data, long length, float maxAmplitude, int offset, double factor) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < length) {
        // Inefficiently find the maximum absolute value within this block
        // For large arrays, a parallel reduction in a separate kernel is recommended.
        float maxVal = 0.0f;
        for (long i = 0; i < length; ++i) {
            if (abs(data[i]) > maxVal) {
                maxVal = abs(data[i]);
            }
        }

        // Calculate the scaling factor
        float scaleFactor = (maxVal > 0.0f) ? (maxAmplitude / maxVal) : 0.0f;

        // Normalize the current element
        data[idx] = data[idx] * scaleFactor;
    }
}
float gaussianKernelPower(float z2, float k) {
    if (k == 1.0) {
        return z2;
    } else if (k == 2.0) {
        return z2 * z2;
    }
    return pow(max(0.0, z2), k);
}

float gaussianKernelScale(float maxStdDev, float k) {
    if (k == 1.0) {
        return maxStdDev;
    } else if (k == 2.0) {
        return sqrt(maxStdDev);
    }
    return pow(max(0.0, maxStdDev), 1.0 / k);
}

float gaussianKernel(float z2, float k) {
    return exp(-0.5 * gaussianKernelPower(z2, k));
}

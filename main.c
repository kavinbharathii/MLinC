#include <stdio.h>
#include <stdlib.h>

float train[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8},
};

float test[][2] = {
    {10, 20},
    {32, 64},
    {99, 198},
    {2143, 4286},
    {11, 22},
};

#define trainCount (sizeof(train) / sizeof(train[0]))

float randomFloat(void) {
    return ((float) rand() / (float) RAND_MAX);
}

float costFunction(float w) {

    float result = 0.0f;

    for (int i = 0; i < trainCount; i++) {
        float x = train[i][0];
        float y = x * w;
        float error = y - train[i][1];
        result += error*error;
    }

    result /= trainCount;
    return result;
}

float derivativeOf(float (*cost)(float), float w) {
    float epsilon = 1e-3;
    return (cost(w + epsilon) - cost(w)) / epsilon;
}

void modelPerformance(float w) {
    for (int i = 0; i < trainCount; i++) {
        float x = test[i][0];
        float y = x * w;
        
        printf("Actual: %d, Predicted: %d\n", (int)test[i][1], (int)y);
    }
}

void main() {
    srand(100);
    float w = randomFloat()*10.0f;
    float derivative;
    float cost;
    float learningRate = 1e-3;
    int steps = 0;

    while (steps < 1000) {
        cost = costFunction(w);
        derivative = derivativeOf(costFunction, w);
        w -= (learningRate * derivative);
        steps += 1;

    }
    printf("%f\n", w);
    modelPerformance(w);
}

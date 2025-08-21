
#include "opengl.h"
#define KERNEL_DIM(x, y) <<<x, y>>>

__device__ Vec2f getBallVelocity(Ball& ball, float dt)
{
    Vec2f newPos = (ball.prevPos * -1.0f) + (ball.currPos * 2.0f) + (ball.acceleration * dt * dt);
    Vec2f velocity = (newPos - ball.prevPos) / (2.0f * dt);
    return velocity;
}

__device__ void setBallVelocity(Ball& ball, Vec2f velocity, float dt)
{
    ball.prevPos = ball.currPos - (velocity * dt) + (ball.acceleration * 0.5f * dt * dt);
}

__device__ Vec2f getBallVelocity_fromNew(Ball& ball, float dt)
{
    Vec2f newPos = (ball.new_prevPos * -1.0f) + (ball.new_currPos * 2.0f) + (ball.acceleration * dt * dt);
    Vec2f velocity = (newPos - ball.new_prevPos) / (2.0f * dt);
    return velocity;
}

__device__ void setBallVelocity_fromNew(Ball& ball, Vec2f velocity, float dt)
{
    ball.new_prevPos = ball.new_currPos - (velocity * dt) + (ball.acceleration * 0.5f * dt * dt);
}

__global__ void wallCollisionKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];
    float dt = simState.dt;

    if (ball.currPos.y - ball.radius < -1.0f) // floor
    {
        Vec2f currVelocity = getBallVelocity(ball, dt);
        Vec2f newVelocity = Vec2f{ currVelocity.x, fabs(currVelocity.y) };
        newVelocity = newVelocity * simState.wallCollisionDampening;
        ball.currPos.y = -1.0f + ball.radius;
        setBallVelocity(ball, newVelocity, dt);
    }

    if (ball.currPos.y + ball.radius > 1.0f) // ceiling
    {
        Vec2f currVelocity = getBallVelocity(ball, dt);
        Vec2f newVelocity = Vec2f{ currVelocity.x, -fabs(currVelocity.y) };
        newVelocity = newVelocity * simState.wallCollisionDampening;
        ball.currPos.y = 1.0f - ball.radius;
        setBallVelocity(ball, newVelocity, dt);
    }

    if (ball.currPos.x - ball.radius < -simState.max_u) // left wall
    {
        Vec2f currVelocity = getBallVelocity(ball, dt);
        Vec2f newVelocity = Vec2f{ fabs(currVelocity.x), currVelocity.y };
        newVelocity = newVelocity * simState.wallCollisionDampening;
        ball.currPos.x = -simState.max_u + ball.radius;
        setBallVelocity(ball, newVelocity, dt);
    }

    if (ball.currPos.x + ball.radius > simState.max_u) // right wall
    {
        Vec2f currVelocity = getBallVelocity(ball, dt);
        Vec2f newVelocity = Vec2f{ -fabs(currVelocity.x), currVelocity.y };
        newVelocity = newVelocity * simState.wallCollisionDampening;
        ball.currPos.x = simState.max_u - ball.radius;
        setBallVelocity(ball, newVelocity, dt);
    }
}

__global__ void ballCollisionKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];
    
    ball.new_currPos = ball.currPos;
    ball.new_prevPos = ball.prevPos;
    ball.wasCrowded = false;

    for (int i = 0; i < balls.size; i++)
    {
        if (i == index) continue;
        Ball& otherBall = balls.devPtr[i];

        Vec2f difference = ball.new_currPos - otherBall.currPos;
        float radiuses = ball.radius + otherBall.radius;
        if (lengthSquared(difference) > (radiuses * radiuses)) continue;
        ball.wasCrowded = true;
        float distance = length(difference);
        if (distance < 0.0000001f) continue;
        
        // elastic collision
        float dt = simState.dt;
        Vec2f V1 = getBallVelocity_fromNew(ball, dt);
        Vec2f V2 = getBallVelocity(otherBall, dt);
        float M1 = ball.mass;
        float M2 = otherBall.mass;
        Vec2f P1 = ball.new_currPos;
        Vec2f P2 = otherBall.currPos;
        Vec2f newVelocity = V1 - ((P1 - P2) * (((2.0f * M2) / (M1 + M2)) * (dot(V1 - V2, P1 - P2) / lengthSquared(P1 - P2))));
        newVelocity = newVelocity * simState.ballCollisionDampening;

        // force apart
        float overlap = radiuses - distance;
        normalize(difference);
        Vec2f directionOffset = difference * overlap;
        ball.new_currPos = ball.new_currPos + directionOffset;
        setBallVelocity_fromNew(ball, newVelocity, dt);
    }
}

__global__ void confirmCollisionKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];

    ball.currPos = ball.new_currPos;
    ball.prevPos = ball.new_prevPos;
}

__global__ void stepPhysicsKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];

    ball.acceleration = Vec2f{ 0.0f, -simState.gravity };
    Vec2f newPos = (ball.prevPos * -1.0f) + (ball.currPos * 2.0f) + (ball.acceleration * simState.dt * simState.dt);
    ball.prevPos = ball.currPos;
    ball.currPos = newPos;
}

__global__ void renderKernel(SimulationState simState)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = -1;
    if ((x < simState.screenWidth) && (y < simState.screenHeight))
        index = y * simState.screenWidth + x;
    else
        return;
    
    float u = ((x / (float)simState.screenWidth) * 2.0f - 1.0f) * (simState.screenWidth / (float)simState.screenHeight);
    float v = (y / (float)simState.screenHeight) * 2.0f - 1.0f;
    Vec2f pixelPos = Vec2f{ u,v };

    uchar4 red = make_uchar4(255, 0, 0, 255);
    uchar4 white = make_uchar4(255, 255, 255, 255);

    SharedArray<Ball>& balls = simState.balls;
    for (int i = 0; i < balls.size; i++)
    {
        Ball& ball = balls.devPtr[i];
        Vec2f relBallPos = ball.currPos - pixelPos;
        if (length(relBallPos) < ball.radius)
        {
            simState.pixels[index] = (ball.wasCrowded) ? red : white;
            return;
        }
    }

    simState.pixels[index] = make_uchar4(0, 0, 0, 255);
    return;
}

void InteropOpenGL::executeKernels(SimulationState& simState)
{
    simState.pixels = nullptr;
    size_t size = 0;
    cudaGraphicsMapResources(1, &cudaPBO, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&simState.pixels, &size, cudaPBO);

    int BALLS_threadsPerBlock = 256;
    int BALLS_blocksPerGrid = (simState.balls.size + BALLS_threadsPerBlock - 1) / BALLS_threadsPerBlock;

    wallCollisionKernel    KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
    ballCollisionKernel    KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
    confirmCollisionKernel KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
    stepPhysicsKernel      KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
    renderKernel           KERNEL_DIM(PIXLS_blocksPerGrid, PIXLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
}


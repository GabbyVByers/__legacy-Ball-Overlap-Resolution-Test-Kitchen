
#include "opengl.h"
#define KERNEL_DIM(x, y) <<<x, y>>>

__device__ bool isClippingWalls(Ball& ball, float max_u)
{
    if (ball.currPos.y - ball.radius < -1.0f)  return true;
    if (ball.currPos.y + ball.radius > 1.0f)   return true;
    if (ball.currPos.x - ball.radius < -max_u) return true;
    if (ball.currPos.x + ball.radius > max_u)  return true;
    return false;
}

__global__ void resolveWallCollisions(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];

    if (ball.currPos.y - ball.radius < -1.0f)
        ball.currPos.y = -1.0f + ball.radius;
    if (ball.currPos.y + ball.radius > 1.0f)
        ball.currPos.y = 1.0f - ball.radius;
    if (ball.currPos.x - ball.radius < -simState.max_u)
        ball.currPos.x = -simState.max_u + ball.radius;
    if (ball.currPos.x + ball.radius > simState.max_u)
        ball.currPos.x = simState.max_u - ball.radius;
}

__global__ void debugCompress(SimulationState simState, float mult)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];

    ball.currPos.y = ball.currPos.y - (0.005f * mult);
}

__global__ void overlapResolutionKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];
    ball.displacement = Vec2f{ 0.0f, 0.0f };
    ball.numFriends = 0;

    for (int i = 0; i < balls.size; i++)
    {
        if (i == index) continue;

        Ball& otherBall = balls.devPtr[i];
        Vec2f difference = ball.currPos - otherBall.currPos;
        float radiuses = ball.radius + otherBall.radius;
        float distance = length(difference);

        if (distance > radiuses) continue;
        if (distance < 0.00001f) continue;

        float overlap = (radiuses - distance);
        normalize(difference);
        ball.displacement = ball.displacement + (difference * (overlap * 0.55f));
        ball.numFriends++;
    }
}

__global__ void applyDisplacementKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];
    //ball.currPos = ball.currPos + (ball.displacement * (1.0f / (float)ball.numFriends));
    if (ball.numFriends != 0)
        ball.currPos = ball.currPos + (ball.displacement / (float)ball.numFriends);
}

__global__ void debugKernel(SimulationState simState)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= simState.balls.size) return;

    SharedArray<Ball>& balls = simState.balls;
    Ball& ball = balls.devPtr[index];

    float max_u = simState.max_u;
    ball.isClipping_Wall = isClippingWalls(ball, max_u);

    ball.isClipping_Ball = false;
    for (int i = 0; i < balls.size; i++)
    {
        if (ball.isClipping_Ball == true) break;
        if (i == index) continue;
        Ball& otherBall = balls.devPtr[i];

        float radiuses = ball.radius + otherBall.radius;
        Vec2f difference = ball.currPos - otherBall.currPos;
        float distance = length(difference);
        if (distance > radiuses) continue;
        ball.isClipping_Ball = true;
    }
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
    uchar4 blue = make_uchar4(0, 0, 255, 255);
    uchar4 purple = make_uchar4(255, 0, 255, 255);
    uchar4 white = make_uchar4(255, 255, 255, 255);

    SharedArray<Ball>& balls = simState.balls;
    for (int i = 0; i < balls.size; i++)
    {
        Ball& ball = balls.devPtr[i];
        Vec2f relBallPos = ball.currPos - pixelPos;
        if (length(relBallPos) < ball.radius)
        {
            uchar4 color = white;
            if (ball.isClipping_Ball) color = red;
            if (ball.isClipping_Wall) color = blue;
            if (ball.isClipping_Ball && ball.isClipping_Wall) color = purple;
            simState.pixels[index] = color;
            //simState.pixels[index] = ball.color;
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


    simState.oddEven++;
    for (int i = 0; i < (128 + (simState.oddEven % 2)); i++)
    {
        resolveWallCollisions   KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
        overlapResolutionKernel KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
        applyDisplacementKernel KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
    }

    debugKernel   KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
    renderKernel  KERNEL_DIM(PIXLS_blocksPerGrid, PIXLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
    debugCompress KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState, 1.0f); cudaDeviceSynchronize();
}

void InteropOpenGL::initImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.FontGlobalScale = 2.0f;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

void InteropOpenGL::renderImGui(SimulationState& simState)
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    ImGui::Begin("Debugger");
    ImGui::Button("Resolve Wall Collision");
    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    {
        int BALLS_threadsPerBlock = 256;
        int BALLS_blocksPerGrid = (simState.balls.size + BALLS_threadsPerBlock - 1) / BALLS_threadsPerBlock;
        resolveWallCollisions KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
        std::cout << "HERE1\n";
    }

    ImGui::Button("Resolve Overlaping");
    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    {
        int BALLS_threadsPerBlock = 256;
        int BALLS_blocksPerGrid = (simState.balls.size + BALLS_threadsPerBlock - 1) / BALLS_threadsPerBlock;
        overlapResolutionKernel KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
        applyDisplacementKernel KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState); cudaDeviceSynchronize();
        std::cout << "HERE2\n";
    }

    ImGui::Button("Debug Compress");
    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    {
        int BALLS_threadsPerBlock = 256;
        int BALLS_blocksPerGrid = (simState.balls.size + BALLS_threadsPerBlock - 1) / BALLS_threadsPerBlock;
        debugCompress KERNEL_DIM(BALLS_blocksPerGrid, BALLS_threadsPerBlock)(simState, 5.0f); cudaDeviceSynchronize();
        std::cout << "HERE2\n";
    }

    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}


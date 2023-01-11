#include <mpi.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    // Initialize MPI
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Read the image
    Mat image = imread("lena.png");
    if (!image.data)
    {
        cout << "Could not open or find the image.\n";
        return -1;
    }
    int rows = image.rows;
    int cols = image.cols;
    // Divide the image into blocks
    int rows_per_block = ceil((double)rows/ num_procs);
    int start_row = rows_per_block * rank;
    int end_row = start_row + rows_per_block;
    Mat block(rows_per_block, image.cols, image.type());
    Mat block_blurred(rows_per_block, image.cols, image.type());

    // Create an MPI datatype for the image block
    MPI_Datatype block_type;
    MPI_Type_vector((end_row - start_row)*3, image.cols, image.rows, MPI_UNSIGNED_CHAR, &block_type);
    MPI_Type_commit(&block_type);

    // Copy the image block to the block variable
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = 0; j < cols*3; j++)
        {
            block.at<uchar>(i - start_row, j) = image.at<uchar>(i, j);
        }
    }

    // Apply the blur filter to the image block
    blur(block, block_blurred, Size(10, 10));
    Mat final_image;
    if (rank == 0)
    {
        final_image = Mat(image.rows, image.cols, image.type());
    }
   MPI_Gather(block_blurred.data, 1, block_type, final_image.data, 1, block_type, 0, MPI_COMM_WORLD);

    // Create a window and display the image
    if (rank == 0)
    {
        namedWindow("bat", WINDOW_AUTOSIZE);
        imshow("bat", final_image);
        waitKey(0);
    }

    // Finalize MPI
    MPI_Type_free(&block_type);
    MPI_Finalize();

    return 0;
}

// za komajliranje komanda: mpic++ main.cpp -o output -lm `pkg-config --cflags --libs opencv4` -lstdc++
// za pokretanje  mpiexec -np 4 --bind-to hwthread ./output
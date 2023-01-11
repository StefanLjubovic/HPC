#include <mpi.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <iostream>

using namespace cv;
using namespace std;



string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main(int argc, char *argv[])
{
    // Initialize MPI
    int rank, num_procs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Read the image
    Mat image = imread("lena.png", IMREAD_UNCHANGED);
    if (!image.data)
    {
        cout << "Could not open or find the image.\n";
        return -1;
    }
    if(rank ==0)cout<<type2str(image.type());
    int rows = image.rows;
    int cols = image.cols;
    int channels = image.channels();

    // Divide the image into blocks
    int count = rows / 2;
    int blocklen = cols * channels / (num_procs / 2); // Each pixel is 3 unsigned chars
    int stride = cols * channels; // Each pixel is 3 unsigned chars

    Mat block(rows / 2, cols / (num_procs / 2), image.type());
    Mat block_blurred(rows / 2, cols / (num_procs / 2), image.type());

    // Create an MPI datatype for the image block
    MPI_Datatype block_type;
    MPI_Type_vector(count, blocklen, stride, MPI_UNSIGNED_CHAR, &block_type);
    MPI_Type_commit(&block_type);

    if (rank == 0)
    {
        // Send the image blocks to the other processes
        for (int i = 1; i < num_procs; i++)
        {
            MPI_Send(&image.data[(i - 1) * count * blocklen], 1, block_type, i, 0, MPI_COMM_WORLD);
        }

        // Process the first image block
        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < blocklen; j++)
            {
                block.at<unsigned short>(i, j) = image.at<unsigned short>(i, j);
            }
        }
        blur(block, block_blurred, Size(10, 10));
    }
    else
    {
        // Receive the image block from the root process
        MPI_Recv(&block.data[0], 1, block_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        blur(block, block_blurred, Size(10, 10));
    }

        Mat final_image;
    if (rank == 0)
    {
    final_image.create(rows, cols, image.type());
    }
        int recvcounts[num_procs];
    int displs[num_procs];

    if (rank == 0)
    {
    // Set the receive counts and displacement arrays
    for (int i = 0; i < num_procs; i++)
    {
    recvcounts[i] = 1;
    displs[i] = i;
    }
    }

    MPI_Gatherv(block_blurred.data, 1, block_type, final_image.data, recvcounts, displs, block_type, 0, MPI_COMM_WORLD);

    // Display the final image
    if (rank == 0)
    {
    namedWindow("Final Image", WINDOW_AUTOSIZE);
    imshow("Final Image", final_image);
    waitKey(0);
    }

    MPI_Type_free(&block_type);
    MPI_Finalize();

    return 0;
}

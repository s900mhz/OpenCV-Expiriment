using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using System.Drawing;

namespace EmguProject
{
    class Program
    {
        #region Constants
        private const string BackgroundFrameWindowName = "Background Frame";

        private const int Threshold = 5; //Determines boundary of brightness while turning grayscale image to binary (black-white) image

        private const int ErodeIterations = 3; //Erosion to remove noise (reduce white pixel zones)
        #endregion

        #region Mats
        private static Mat backgroundFrame = new Mat(); // Frame used as a base for change detection 

        private static Mat rawFrame = new Mat(); // Frame as is obtained from video
        private static Mat diffFrame = new Mat(); //Image showing differences between background and raw frame

        private static Mat grayscaleDiffFrame= new Mat(); //Image showing differences in 8-bit color depth
        private static Mat binaryDiffFrame = new Mat(); //Image showing changes areas in white and unchanges in black

        private static Mat denoisedDiffFrame = new Mat(); // Image with irrelevant changes removed with opening operation

        private static Mat finalFrame = new Mat(); //Video frame with detected object marked

        #endregion


        static void Main(string[] args)
        {
            string videoFile = @"C:\Users\Sean Spade\Videos\emgu_cv_drone_test_video.mp4";

            using(var capture = new VideoCapture(videoFile)) //Loading video from file
            {
                if (capture.IsOpened)
                {
                    // Obtaining and showing first frame of loaded video 
                    // (used as the base for difference detection)
                    backgroundFrame = capture.QueryFrame();
                    CvInvoke.Imshow(BackgroundFrameWindowName, backgroundFrame);

                    // Handling video frames (image processing and contour detection)
                    VideoProcessingLoop(capture, backgroundFrame);
                }
                else
                {
                    Console.WriteLine($"Cannot open {videoFile}");
                }

            }


        }
        private static void VideoProcessingLoop(VideoCapture capture, Mat backgroundFrame)
        {
            var stopwatch = new Stopwatch(); // Used for measuring video proccesing performence

            int frameNumber = 1;
            while(true) //Loop video
            {
                Mat rawFrame = capture.QueryFrame(); //grabs next frame. Null is returned if no further frame exists

                if (rawFrame != null)
                {
                    frameNumber++; //Moves the counter to the next frame

                    stopwatch.Restart(); //restarts the stopwatch
                    ProcessFrame(backgroundFrame, Threshold, ErodeIterations, DilateIterations);
                    stopwatch.Stop(); //stops the watch so we know how long it took to proccess the frame

                    WriteFrameInfo(stopwatch.ElapsedMilliseconds, frameNumber);
                    ShowWindowsWithImageProcessingStages();

                    int key = CvInvoke.WaitKey(0); //waits indefinitely until key is pressed

                    if (key == 27)
                    {
                        Environment.Exit(0);
                    }
                }
                else
                {
                    capture.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.PosFrames, 0);//Moves to first frames
                    frameNumber = 0;
                }
            }
        }

        private static void ProcessFrame(Mat backgroundFrame, int threshold, int erodeIterations, int dilateIterations)
        {
            //Find difference between background (first) frame and current frame
            CvInvoke.AbsDiff(backgroundFrame, rawFrame, diffFrame);

            //Apply binary threshold to grayscale image (white pixel will mark difference)
            CvInvoke.CvtColor(diffFrame, grayscaleDiffFrame, ColorConversion.Bgr2Gray);
            CvInvoke.Threshold(grayscaleDiffFrame, binaryDiffFrame, threshold, 255, ThresholdType.Binary);


            //Remove noise with opening operation (erosion followed by dilation)
            CvInvoke.Erode(binaryDiffFrame, denoisedDiffFrame, null, new Point(-1, -1), erodeIterations,
                            BorderType.Default, new Emgu.CV.Structure.MCvScalar(1));
            CvInvoke.Dilate(denoisedDiffFrame, denoisedDiffFrame, null, new Point(-1, -1), dilateIterations,
                            BorderType.Default, new Emgu.CV.Structure.MCvScalar(1));

            rawFrame.CopyTo(finalFrame);
            DetectObject(denoisedDiffFrame, finalFrame);
        }
        
    }
}

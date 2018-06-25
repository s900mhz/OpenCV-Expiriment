using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Util;

namespace EmguProject
{
    class Program
    {
        private const string BackgroundFrameWindowName = "Background Frame";

        private static Mat backgroundFrame = new Mat(); // Frame used as a base for change detection 

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
    }
}

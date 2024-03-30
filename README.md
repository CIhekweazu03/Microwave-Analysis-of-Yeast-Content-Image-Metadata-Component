# Preprocessing for Cell Image/Video Analysis
## sf_video_clipper.py
The current script for processing videos based on the format seen in the box folder https://clemson.app.box.com/folder/235122244224
It takes an input folder and iterates over all the subfolders using the ".mat" file to get the correct video and timestamp for each cell as well as keeping track of the cell ids.
### CLI Options
Currently, the code uses the Python click module for a CLI; the options for the CLI are below, only two are required (input directory and output directory), but the rest of the options may need to be tweaked for more accurate processed videos. 

-dd, --input_directory TEXT     Input directory that contains the 'set'
                                  subfolders and videos under the format
                                  'video_log.mp4'  [required]

                      
-od, --output_directory TEXT    Directory that output will be saved to under
                                the format of '{input_directory}/{output_dir
                                ectory}/Saved_Clips_{method}'  [required]

                                
-m, --method TEXT               Method that you want for the video to be
                                processed with

                                
-d, --duration INTEGER          Duration of the clips for the cells that
                                will be looked at. Example, 1 second
                                duration means get 0.5 seconds before
                                timestamp and 0.5 seconds after timestamp.

                                
-t, --tolerance FLOAT           Tolerance 'threshold' for what is to be
                                considered a similar frame. Example, 10%
                                similar frame will be 0.1 and mean that you
                                consider frames with a mean that are within
                                +/- 10% similar to one another.

                                
-tv, --threshold_value INTEGER  Threshold value for thresholding images;
                                above this value will be white, below black

                                
-ef, --extract_frames_flag      Save all of the frames of the outputted
                                clips into a subfolder

                                
--help                          Show this message and exit.

## Next Steps
* With the lighting not being an issue in future videos, experiment with simply averaging each pixel’s values for the video rather than the current process of finding all similar frames and subtracting the average for all similar frames.
* Video stabilization with a module such as 'vidstab.'
* Cropping the video before any processing (currently tricky due to inconsistency with sensor location).

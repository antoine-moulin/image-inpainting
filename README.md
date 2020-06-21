<h1> README </h1> 

<h2> Image Inpainting </h2>

Image inpainting makes it possible to erase elements present in an image and replace them with a plausible background, in particular by reproducing textures when the area to be filled is relatively large and by propagating linear structures such as contours.

This project has been coded by [Jean Vassoyan](https://github.com/jvasso) and [Antoine Moulin](https://github.com/moulinantoine/) - students at Télécom Paris - under the supervision of [Alasdair Newson](https://sites.google.com/site/alasdairnewson/). It is based on the method described in this [paper](http://www.irisa.fr/vista/Papers/2004_ip_criminisi.pdf).

There are several types of files in this projet :
- the GUI files, which contain a python program to run a GUI
- the class files, which contain a python class
- the dataset directory which contains a few folders with data on images
- other files : <tt>useful_functions.py</tt> and <tt>Inspect_Data.py</tt>



<h2> The GUI Files </h2>

There are two GUI files : <tt>GUI.py</tt> and <tt>GUI_for_tests.py</tt>. The first one makes it possible to make an inpainting on a single image by drawing a mask on it or loading it from your own directory. However, no data are recorded about this inpainting and you cannot compare statistical perormance. The second one makes it possible to record data about an inpainting made on a whole dataset of images. However you cannot draw your own masks with, to work properly, it requires a file named "mask.jpg" in the folder of each image of the dataset.

**WARNING** : to use the program properly, you have to run one of the GUI files and follow the instructions. Please make sure you do have these python libraries before you start the program :
- numpy
- scipy
- sklearn
- cv2
- tkinter
- PIL
- pickle
- matplotlib

**It is also important to have GIMP** on your computer and check that the calls defined in files "useful_functions" do start GIMP on your laptop. Otherwise you will not be able to display images during the inpainitng.

In order to use <tt>GUI.py</tt> : 
1. click on Browse to search your image
2. draw a mask on the image (or load one if it already exists)
3. choose the parameters (if the number of clusters equals 1, the algorithm will use a restricted search area)
4. choose the frequency of the display
5. click on Start inpainting

In order to use <tt>GUI_for_tests.py</tt> :
1. click on Browse to search the folder containing your data set
2. select the parameter you want to change during the tests, and its range
3. select the others parameters
4. write a short description of the test
5. click on Start the test 



<h2> The class Files </h2>

There are 3 class files : <tt>Image.py</tt>, <tt>Pixel.py</tt>, <tt>Test.py</tt>.

  <tt>Image.py</tt> contains all the methods necessary to make an inpainting.  
  <tt>Pixel.py</tt> representents a pixel in the image and is used by <tt>Image.py</tt>.  
  <tt>Test.py</tt> is a class that representents a Test made on one image. It is used by the <tt>GUI_for_tests.py</tt> file.  



<h2> The Dataset directory </h2>

The "dataset" directory is used by the <tt>GUI_for_tests.py</tt> file. It contains a set of folders which are called "data1", "data2", "data3" etc. In each data folder, you can find a file called "image.jpg", another one called "mask.jpg" and a directory called "tests" which contains python objects from the <tt>Test.py</tt> class. (Those objects were saved thanks to the module pickle)
It is very important to follow this rule about the names of image and mask in data directories. Otherwise the program cannot find them.


<h2> The other files </h2>

<tt>useful_functions.py</tt> : it contains two "viewimage" functions which call GIMP with specific commands. Please check that these commands are adapted to your laptop before running the program.  

<tt>Inspect_Data.py</tt> : it contains a few functions to explore the data genarated with the tests made with <tt>GUI_for_test.py</tt>

<h2> An example </h2>

Here is an example of inpainting obtained using our GUI:

![results/selfie_before.jpg](results/selfie_before.jpg)
![results/selfie_after.jpg](results/selfie_after.jpg)

**NB** :

- if you want to do an inpainting without using any optimization method you can just chose method number 1 (clustering on pixels) with a number of clusters equal to 1

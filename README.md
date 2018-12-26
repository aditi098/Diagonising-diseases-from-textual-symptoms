To run the code:

1. Train the autoencoder and classifiers. Run the code.py file in python3.

    $ python3 code.py

2. Test using file test.py.

    $ python3 test.py

Other than the 2 code files, 4 text input files are also included:

1. malaria.txt

2. viralfever.txt

3. dengue.txt

4. typhoid.txt

Description of other files/directories:

1. weights: Directory- contains saved autoencoder model

2. log2: Directory- contains logs for neural network training

3. Classifier trained models in .sav files:

     - decision_tree.sav

     - Kmeans.sav

     - naive_bayes.sav

     - random_forest.sav

     - regression.sav

     - svm.sav

4. Input_script.c- C script to generate input files as mentioned above.

	 - To run this file:

	 	1. Change the filename to disease.txt in  the line *FILE *fptr= fopen("disease.txt", "w");*, where disease can be typhoid, malaria, dengue, viralfever

	 	2. Select the appropriate array of symptoms from the commented code. Comment all other symptoms.

	 	3. In terminal, run the C file:

	 		- $ gcc -c Input_script.c

	 		- $ gcc Input_script.o -o Input_script

	 		- $ gcc ./Input_script

	 	4. Run the file 4 times for each of the disease, to generate 4 text input files.
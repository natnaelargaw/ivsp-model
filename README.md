# Overview
		This repostitory holds a code base for the research work a dynamic video saliency identification, which is still a work in progress. 
		It is mainly inspired by the attentive CNN-LSTM based architecture proposed in [1]. We are improving its performance by tweaking its preprocessing layer/s and the overall architecture. 
    

# Imprint

		Paper: Interactive Audio-Visual Saliency Identification
		Dataset: DHF1K
		State of the Art: ACL




# Dependencies
**Name**                 							   **Version**                  
keras                                  2.4.3                       
matplotlib                       3.5.1            
numpy                              1.22.3        
opencv-python             4.5.5.64            
pillow                                9.1.1                   
scipy                     			   1.7.3          
tensorflow                        2.4.1         
tensorflow-estimator      2.9.0                
tensorflow-gpu            2.4.1           
tensorflow-hub            0.12.0              
tornado                           6.1              
yaml                                  0.2.5              


# Folder structure
		repo --> audiovisual_saliency
		Video Folders
		Training (DHF1K 001 - 600 videos)  Location: repo   ./DHF1K/training/
		Validation (DHF1K 601 - 700 videos) Location: repo ./DHF1K/validation/
		Test       (DHF1K 701- 1000 videos) Location: repo ./DHF1K/test/


# Procedures
- First run the extract_frames.py python file to extract video frames. 
		This file puts frames in their corresponding video name directory as shown above
- To train
		 In the main.py make sure the phase_gen variable is set to 'train'.
		 Check if the file paths after the completion of extract_frames.py is correct
		 run main.py
		  localhost:6606/learner or IP:6606/learner is a web based interface to interactively monitor the learning model and agent view in real time.

- To tesT
		In the main.py file, change the phase_gen to 'test'
		run the main.py file
		 localhost:6606/learner or IP:6606/learner is a web based interface to interactively monitor the learning model and agent view in real time.

# Outputs
		For the training phase,  it saves epoch level chekpoints in the same folder as main.py.
		For the testing phase, it generates prediction maps to corresponding test folder under the name 'saliency'
		 A web based realtime saliency monitoring interface -- > localhost:6606/learner or IP:6606/learner

# Task in progress
		We are enhancing the model performance by using some preprocessing and parameter tuning techniques. 

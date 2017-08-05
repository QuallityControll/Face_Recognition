# Face-Recognition

## Recognizes faces - 
If the person is not in the database: returns "I see one person I dont recognize" or "I see ____ people I dont recognize".

If the person/people are in the database: returns "I see (name)" or "I see (names)".

## Installation Instuctions:

1) Please install the camera from the following link https://github.com/LLCogWorks2017/Camera * and use the instructions from the link.

2) Clone repository into desired file 

3) Run setup.py files to run face recognition use following command:

      ```python fr_setup.py develop```
      
4) Use the command to start the face recognition package after running the setup.py file (use the following command): 
     
     ```import face_recognition as fr```

### \*Credits

Camera

The camera was imported from [Camera](https://github.com/LLCogWorks2017/Camera), developed by [Ryan Soklaski](https://github.com/LLrsokl), lead instructor for the CogWorks 2017 Summer Program at MIT. 

Camera was created for the [Beaver Works Summer Institute at MIT](https://beaverworks.ll.mit.edu/CMS/bw/bwsi)

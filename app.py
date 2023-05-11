import numpy as np
from flask import Flask,request,render_template
import pickle

#create an app object using flask class
app=Flask(__name__)
#Load the trained model. (Pickle file)

model=pickle.load(open('./model/model.pkl','rb'))

#Define the route to be home.

#The decorator below Links the relative route of the URL to the function it is decc 
# #Here, home function is with /, our root directory. 
# #Running the app sends us to index.html. 
# #Note that render template means it looks for the file in the templates folder.

#use the route() decorator to tell Flask what URL should trigger our function.
@app.route('/')
def home():
    return render_template('index.html')




#You can use the methods argument of the route() decorator to handle different HTTP methods
# #GET: A GET message is send, and the server returns data
#POST: Used to send HTML form data to the server.
#Add Post method to decorator to allow for form submission
#Redirect to /predict page with the output


@app.route('/predict',methods=['POST'])
def predict():
    # from sklearn.feature_extraction.text import CountVectorizer
    # cv=CountVectorizer()
    from model import cv
    ip=""
    for x in request.form.values():
        ip+=x
    inp=[x for x in request.form.values()]
    inp=cv. transform(inp). toarray()
    output=model.predict(inp)
    return render_template('index.html',prediction_text=output[0],ip=ip)


#when the Python interpreter reads a source file, it first defines a few special variable
#For now, we care about the name variable.
#If we execute our code in the main program, Like in our case here, it assigns 
# #_main_ as the name (_name_).
#So if we want to run our code right here, we can check if __name__ == __main_ 
# #if so, execute it here.


#If we import this file (module) to another file then __name__ == app(which is name of this python file)


if __name__=="__main__":
    app.run(debug=True)
    
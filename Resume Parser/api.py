from fastapi import FastAPI, UploadFile, File
import pickle
import pdfplumber
import re
import nltk
from nltk.corpus import stopwords

app = FastAPI()

classifier = pickle.load(open('pickles/classifier.pkl', 'rb'))
vectorizer = pickle.load(open('pickles/vectorizer.pkl', 'rb'))

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def pdf_to_text(pdf):
    text = ""
    with pdfplumber.open(pdf) as pd:
        for page_number in range(len(pd.pages)):
            page = pd.pages[page_number]
            text += page.extract_text()
    return text

def process_text(text):
    cleanTxt = re.sub('https://\S+', ' ', text)
    cleanTxt = re.sub('www.\S+', ' ', cleanTxt)
    cleanTxt = re.sub('@\S+', ' ', cleanTxt)
    cleanTxt = re.sub('#\S+', ' ', cleanTxt)
    cleanTxt = re.sub('[%s]' % re.escape("""!@#$%^&*()~`{}[]:;"'<,>.?/|\-="""), ' ', cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]', ' ', cleanTxt)
    cleanTxt = re.sub('\s+', ' ', cleanTxt)
    cleanTxt = [w for w in cleanTxt.split() if not w.lower() in stop_words]
    cleanTxt = ' '.join(cleanTxt)
    vectored_text = vectorizer.transform([cleanTxt])
    return vectored_text

def prediction(data):
    prediction_id = classifier.predict(data)[0]
    classes_encoding = {0: 'Advocate',
                        1: 'Arts',
                        2: 'Automation Testing',
                        3: 'Blockchain',
                        4: 'Business Analyst',
                        5: 'Civil Engineer',
                        6: 'Data Science',
                        7: 'Database',
                        8: 'DevOps Engineer',
                        9: 'DotNet Developer',
                        10: 'ETL Developer',
                        11: 'Electrical Engineering',
                        12: 'HR',
                        13: 'Hadoop',
                        14: 'Health and fitness',
                        15: 'Java Developer',
                        16: 'Mechanical Engineer',
                        17: 'Network Security Engineer',
                        18: 'Operations Manager',
                        19: 'PMO',
                        20: 'Python Developer',
                        21: 'SAP Developer',
                        22: 'Sales',
                        23: 'Testing',
                        24: 'Web Designing'}
    predicted_category = classes_encoding[prediction_id]
    return predicted_category

@app.post("/predict")
async def root(file: UploadFile = File(...)):

    text = pdf_to_text(file.file)
    vectorized_text = process_text(text)
    resume_category = prediction(vectorized_text)

    return {"resume_category": resume_category}

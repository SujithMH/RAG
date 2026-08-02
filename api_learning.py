from fastapi import FastAPI,Path
from pydantic import BaseModel

app= FastAPI()
student={
    1:{"name" :"sujith",
       "age":21,
       "course":"data science"
    }
}
class Student(BaseModel):
    name:str
    age:int
    course:str

class update_student(BaseModel):
    name :str=None
    age:int =None
    course:str=None

@app.get("/")
def home():
    return {"message":"hello world"} 

@app.get("/students-data/{student_id}")
def student_data(student_id:int = Path(...,description="enter the id of the student you want to see",gt=0,lt=3)):
    return student[student_id]

@app.get("/get_by_name")
def stud_data(*,student_id:int=None,name:str,test:int=None):   # here none specifies thet its not a compulsory value ,* specifies multiple default pamaeters to initialize them
    for student_id in student:
        if student[student_id]["name"]==name:
            return student[student_id]
        
        return {"data":"not found"}

@app.post("/create_student/{student_id}")
def add(student_id:int,student_data:Student):
    if student_id in student:
        return "error"
    
    student[student_id]=student_data.model_dump()
    return student[student_id]

@app.put("/update_student/{student_id}")
def update(student_id:int,student1:update_student):
    if student_id not in student:
        return {"error"}
    student[student_id]=student1
    return student[student_id]

@app.delete("/delete")
def delete(student_id:int):
    if student_id not in student:
        return "error does not exist"
    del student[student_id]
    return {"message": f"{student_id} deleted"}
    
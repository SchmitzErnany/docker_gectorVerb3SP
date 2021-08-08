# Use an official Python runtime as an image
FROM python:3.6-slim

# The EXPOSE instruction indicates the ports on which a container 
# will listen for connections
# Since Flask apps listen to port 5000 by default, we expose it
EXPOSE 8000

# Sets the working directory for following COPY and CMD instructions
# Notice we haven’t created a directory by this name - this instruction 
# creates a directory with this name if it doesn’t exist

WORKDIR /

RUN mkdir app1

WORKDIR /app1

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app1
RUN apt-get update && apt-get install -y build-essential && pip3 install --upgrade pip && pip3 install -r requirements.txt

# Run app.py when the container launches
COPY  .  /app1/
ENV min_err_prob=0.7 add_conf=0.3 tokenizer_method=split
CMD ["python3","manage.py","runserver","--noreload","0.0.0.0:8000"]

FROM centos:latest

RUN pip3 install --upgrade pip
RUN pip3 install sklearn
RUN pip3 install numpy
RUN pip3 install pandas

CMD ["python3","/home/arundsp/ws/mlopsfinalproj/task.py"]



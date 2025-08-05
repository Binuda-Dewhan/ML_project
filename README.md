## End to end machine learning project 

## Add venv environment using 
conda create -p venv python=3.10 -y

and activate 

conda activate venv

conda activate "D:\Personal Project_certificate work Space(Projects )\ML_project_repo\venv"


## install libraries

pip install -r requirments.txt


## CI/CD 

to activate the ec2

### at the initial
icacls .\flask-key.pem /inheritance:r
icacls .\flask-key.pem /grant:r "${env:USERNAME}:(R)"

ssh -i "flask-key.pem" ubuntu@<public ipv4 address> - this can use after initiate

sudo apt update && sudo apt upgrade -y 

### create virtual env
sudo apt install python3 python3-pip python3-venv git -y

check python version - python3 --version
check pip version - pip3 --version

### cloning git repo 
 git clone https://github.com/Binuda-Dewhan/ML_project.git

### go to the repo folder 
cd ML_project
python3 -m venv venv

activate virtual environment - source venv/bin/activate

install requirement (libraries) - pip install -r requirements.txt

## run the application 
python app.py


## run the app.py after closing (optional) - with nohup

### Run Flask in the Background with nohup
To keep Flask running even after closing SSH:

nohup python application.py &     - this will runs in background, and output goes to nohup.out

### To Stop the Flask Server
If it's running in foreground, press CTRL + C.

### If it's running in background, Find the process ID (PID) and stop it:
ps aux | grep python 

kill it 
kill <PID>








import subprocess
import time



# CSV files names
f1 = "../dynamic_results/f1.csv"
f2 = "../dynamic_results/f2.csv"

# Commands to execute
c0 = "idf.py -p COM3 monitor"
c1 = 'findstr "CSI_DATA" > ../dynamic_results/f1.csv'
c2 = 'findstr "CSI_DATA" > ../dynamic_results/f2.csv'

while True:
    try:
        # Wait up to 10 seconds for the second command to finish executing
        max_execution_time = 10  # seconds

        # Execute the first command with pipe
        process0 = subprocess.Popen(c0, shell=True, stdout=subprocess.PIPE)

        start_time1 = time.time()
        while time.time() - start_time1 < max_execution_time:
            process1 = subprocess.Popen(c1, shell=True, stdin=process0.stdout)
            time.sleep(0.1)  # Short wait to avoid overloading the processor

        process1.kill()


        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("                                                                                         ")
        print("++++++++++++++++++++++++++++++++++++++1111+++++++++++++++++++++++++++++++++++++++++++++++")
        print("                                                                                         ")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


        start_time2 = time.time()
        while time.time() - start_time2 < max_execution_time:
            process2 = subprocess.Popen(c2, shell=True, stdin=process0.stdout)
            time.sleep(0.1)  # Short wait to avoid overloading the processor

        process2.kill()


        print("-----------------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------------")
        print("                                                                                         ")
        print("----------------------------------------2222---------------------------------------------")
        print("                                                                                         ")
        print("-----------------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------------")
        

        process0.kill()

        print("Commands executed successfully.")
    except Exception as e:
        print("An error has occurred:", str(e))

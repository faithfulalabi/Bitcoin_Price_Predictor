# Bitcoin_Price_Predictor

* A Machine Learning(ML) model that predicts if Bitcoins price will go up or down 7 days from now using its basic historical data to be better than dollar-cost averaging. 
* [Check it out](http://futurecryptoprice.io/)
* Check out the architecture
* Read below for motivation, how to rebuild it locally, and model performance 

## Motivation :
I wanted to build an end-to-end ML project that didn't just live locally on my laptop. I also wanted to develop a Web3 data-related product, so I chose to provide an answer to a question I had, which was how can I be better at [dollar-cost averaging](https://www.investopedia.com/terms/d/dollarcostaveraging.asp#:~:text=Dollar%2Dcost%20averaging%20(DCA)%20is%20an%20investment%20strategy%20in,price%20and%20at%20regular%20intervals.) Bitcoin? 


## Rebuilding Locally:
* Install [PgAdmin](https://www.pgadmin.org/) or any othe Postgres Clinet
* Clone the repo to your local directory
* Activate a conda or python environment 
* Pip install requirements.txt
* Setup your PgAdmin Server
* Create a Database(DB) called 'crypto'
* create a table called 'bitcoin'
* create another table called 'processed_bitcoin'
* go into every folder and add a database.ini file that follows the structure below:
[postgresql]
host=<your_localhost>
database=<name_of_db_created>
user=<your_postgres_user_name>
password=<your_postgres_password>
* Navigate to initial_data folder and run file called run_initial_data.py (This script gets the historical data of bitcoin and stores it in table called bitcoin)
* Navigate to the process_data folder and run file called run_process_data.py (This transform the bitcoin data into the format that the model will be trained on)
* Go to the daily_data folder and run file called run_daily_request.py everyday or set up a cron job that'll automate the daily run
* Lastly go to the app and run this command 'streamlit run main.py' and you should see it live locally! Hooray!


## Model Performance 
* I set my model accuracy goal as anything % more than 50% since my goal was to be better at dollar-cost averaging. Plus, I didn't want to spend too much time on the modeling since this was an end-to-end project..

Prediction completed.
 Model Accuracy is: 53.5%
 ![image](https://user-images.githubusercontent.com/40880554/160971800-f5c15e2d-ac54-4dfd-8bad-84887d475735.png)





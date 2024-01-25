from codecs import ignore_errors
from pickle import FALSE
import streamlit as st
from streamlit_drawable_canvas import st_canvas

import cv2
import requests
import urllib
import json
import os
import numpy as np
import time

import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
# Configs
MODEL_INPUT_SIZE = 28
CANVAS_SIZE = MODEL_INPUT_SIZE * 8

if os.environ.get("BACKEND_URL") is not None:
    BACKEND_URL = os.environ.get("BACKEND_URL")
else:
    BACKEND_URL = "http://localhost:8000"

GET_JOB_URL = urllib.parse.urljoin(BACKEND_URL, "jobs")
GET_JOB_TRAIN_URL = urllib.parse.urljoin(BACKEND_URL, "jobs_train")
GET_JOB_GRIDSEARCH_URL = urllib.parse.urljoin(BACKEND_URL, "jobs_gridsearch")
TRAIN_URL = urllib.parse.urljoin(BACKEND_URL, "train")
GRIDSEARCH_URL = urllib.parse.urljoin(BACKEND_URL, "grid_search")
UPDATE_TRAIN_URL = urllib.parse.urljoin(BACKEND_URL, "update_train")
UPDATE_GRIDSEARCH_URL = urllib.parse.urljoin(BACKEND_URL, "update_gridsearch")

df = None

st.set_page_config(
    page_title="Experiment Management",
    page_icon="✌️"
)

st.title("Experiment Management System")
st.sidebar.subheader("Experiments")

# menu sidebar
page = st.sidebar.selectbox(label="Menu", options=[
    "Create Job", "Progress", "Display"])

st.sidebar.success("Select an option above")

# create a job
if page == "Create Job":
    # choose kind of job
    selected = option_menu(
        menu_title=None,
        options=["Train", "Grid Search"],
        icons=["house", "book"],
        orientation="horizontal"
    )

    if selected == "Train":
        st.session_state.model_type = st.selectbox(
            "Model type", options=["Linear"])

        # select parameters
        if st.session_state.model_type == "Linear":
            # num_layers = st.select_slider(
            #     label="Number of hidden layers", options=[1, 2, 3])

            # cols = st.columns(num_layers)
            # hidden_dims = [64] * num_layers
            # for i in range(num_layers):
            #     hidden_dims[i] = cols[i].number_input(
            #         label=f"Number of neurons in layer {i}", min_value=2, max_value=128, value=hidden_dims[i])


            epochs = st.number_input("Epochs", min_value=1, value=5, max_value=128)

            learning_rate = st.number_input("Learning Rate", min_value=1e-3, value=2e-3, step=1e-3, max_value=5e-3, format="%.3f")

            drop_out = st.number_input("Drop Out", min_value=0.1, value=0.5, step=0.1, max_value=0.9, format="%.f")

            # hyperparams = {
            #     "hidden_dims": hidden_dims,
            #     "drop_out": drop_out
            # }

        if st.button("Train"):
            to_post = {"drop_out": drop_out,
                       "learning_rate": learning_rate, 
                       "epochs": epochs}

            # send a request to create and start a train job
            response = requests.post(url=TRAIN_URL, data=json.dumps(to_post))

            if response.ok:
                if response.json()["task_id"] == "Already existed !!!":
                    res = "Job " + " already existed !!!"
                else:
                    res = "Job " + response.json()["task_id"] + " started"
            else:
                res = "Training task failed"

            st.write(res)
    else:
        # select parameters
        options_drop_out = st.multiselect(
            'Drop Out',
            [0.3, 0.5, 0.8],
            [0.3, 0.5]
        )


        options_epochs = st.multiselect(
            'Epochs',
            [1, 3, 5, 7, 10],
            [1, 3]
        )

        options_learning_rate = st.multiselect(
            'Learning Rate',
            [1e-3, 2e-3, 5e-3],
            [1e-3, 2e-3]
        )

        if st.button("Grid Search"):
            to_post = {
                "params_dict":{
                    "epochs": options_epochs,
                    "drop_out": options_drop_out,
                    "learning_rate": options_learning_rate
                }
            }

            # send a request to create and start a grid search job
            response = requests.post(url=GRIDSEARCH_URL, data=json.dumps(to_post))

            if response.ok:
                if response.json()["task_id"] == "Already existed !!!":
                    res = "Job " + " already existed !!!"
                else:
                    res = "Job " + response.json()["task_id"] + " started"
            else:
                res = "Training task failed"

            st.write(res)

# track progress of jobs
elif page == "Progress":
    # choose kind of job
    selected = option_menu(
        menu_title=None,
        options=["Train", "Grid Search"],
        icons=["house", "book"],
        orientation="horizontal"
    )

    if selected == "Train":
        response = requests.get(url=GET_JOB_TRAIN_URL)

        res = response.json()

        l = {}

        c = {}

        mark = {}

        for r in res:
            progress_text = f"0/{r['params']['epochs']} Epochs"
            l[r['uid']] = st.progress(0, text="Task " + r['uid'] + " : " + progress_text)
            c[r['uid']] = [st.line_chart(), st.line_chart()]
            mark[r['uid']] = False
            
        
        while True: 
            for r in res:
                response = requests.get(url=GET_JOB_URL + '/' + r['uid'])

                result = response.json()

                # get current state and information of each job
                if result['task_status'] == "PROGRESS":
                    current = result['task_result']['current']
                    total = result['task_result']['total']
                    data = result['task_result']['metrics']

                    loss = {'train_loss': data['train_loss'],
                            'val_loss': data['val_loss']}
                    
                    acc = {'train_acc': data['train_acc'],
                            'val_acc': data['val_acc']}

                    loss_data = pd.DataFrame(loss)

                    acc_data = pd.DataFrame(acc)

                    display = "Task " + r['uid'] + " : " + str(current) + '/' + str(total) + " Epochs"
                    time.sleep(0.01)
                    process_percent = int(100 * float(current) / float(total))

                    # display progress bar
                    l[r['uid']].progress(process_percent, text=display)

                    # plot loss and accuracy
                    c[r['uid']][0].line_chart(loss_data)

                    c[r['uid']][1].line_chart(acc_data)

                elif result['task_status'] == "SUCCESS":
                    data = result['task_result']

                    loss = {'train_loss': data['train_loss'],
                            'val_loss': data['val_loss']}
                    
                    acc = {'train_acc': data['train_acc'],
                            'val_acc': data['val_acc']}

                    loss_data = pd.DataFrame(loss)

                    acc_data = pd.DataFrame(acc)
                    
                    # display progress bar
                    l[r['uid']].progress(100, text="Task " + r['uid'] + " : " + "Done !!!")

                    # plot loss and accuracy
                    c[r['uid']][0].line_chart(loss_data)

                    c[r['uid']][1].line_chart(acc_data)

                    job = {
                            "uid": r['uid'],
                            "type": r['type'],
                            "params": r['params'],
                            "status": "Done",
                            "accuracy": data['val_acc'][-1],
                            "run_time": data['run_time']
                        }

                    response = requests.post(url=UPDATE_TRAIN_URL, data=json.dumps(job))

                    mark[r['uid']] = True

            if all(mark.values()):
                break
    else:
        response = requests.get(url=GET_JOB_GRIDSEARCH_URL)

        res = response.json()

        l = {}

        mark = {}

        for r in res:
            progress_text = f"0/100 %"
            l[r['uid']] = st.progress(0, text="Task " + r['uid'] + " : " + progress_text)
            mark[r['uid']] = False
            
        
        while True: 
            for r in res:
                response = requests.get(url=GET_JOB_URL + '/' + r['uid'])

                result = response.json()
                
                # get current state and information of each job
                if result['task_status'] == "PROGRESS":
                    current = result['task_result']['current']
                    total = result['task_result']['total']
                    process_percent = int(100 * float(current) / float(total))

                    display = "Task " + r['uid'] + " : " + str(process_percent) + '/' + "100" + " %"
                    time.sleep(0.01)
                    
                    # display progress bar
                    l[r['uid']].progress(process_percent, text=display)

                elif result['task_status'] == "SUCCESS":
                    data = result['task_result']

                    # display progress bar
                    l[r['uid']].progress(100, text="Task " + r['uid'] + " : " + "Done !!!")

                    job = {
                            "uid": r['uid'],
                            "type": r['type'],
                            "params": r['params'],
                            "status": "Done",
                            "best_accuracy": data['best_acc'],
                            "best_params": data['best_params'],
                            "run_time": data['run_time']
                        }

                    response = requests.post(url=UPDATE_GRIDSEARCH_URL, data=json.dumps(job))

                    mark[r['uid']] = True

            if all(mark.values()):
                break

# display all finished jobs
elif page == "Display":
    # choose kind of job
    selected = option_menu(
        menu_title=None,
        options=["Train", "Grid Search"],
        icons=["house", "book"],
        orientation="horizontal"
    )

    if selected == "Train":
        # send a request to get all finished train jobs in database
        response = requests.get(url=GET_JOB_TRAIN_URL)
        
        data = response.json()

        uid = []
        epochs = []
        learning_rate = []
        drop_out = []
        accuracy = []
        run_time = []

        for d in data:
            if d['status'] == "Done":
                uid.append(d['uid'])
                learning_rate.append(d['params']['learning_rate'])
                drop_out.append(d['params']['drop_out'])
                epochs.append(d['params']['epochs'])
                accuracy.append(d['accuracy'] * 100)
                run_time.append(d['run_time'])


        dict_val = {"Job ID": uid, 
                    "Learning Rate": learning_rate,
                    "Epochs": epochs,
                    "Drop Out": drop_out,
                    "Accuracy (%)": accuracy,
                    "Run Time (s)": run_time
                    }

        df = pd.DataFrame(dict_val)

        st.dataframe(df, use_container_width=True)
    else:
        # send a request to get all finished grid search jobs in database
        response = requests.get(url=GET_JOB_GRIDSEARCH_URL)
        
        data = response.json()

        for d in data:
            if d['status'] == "Done":
                label = f"Job {d['uid']}"

                s = f"<p style='font-size:20px;'> {label}</p>"
                st.markdown(s, unsafe_allow_html=True)  
                st.write(f"**Grid Search Space**: Epochs: {d['params']['epochs']}, Learning Rate: {d['params']['learning_rate']}, Drop Out: {d['params']['drop_out']}")
                st.write(f"**Best parameters**: Epochs: {d['best_params']['epochs']}, Learning Rate: {d['best_params']['learning_rate']}, Drop Out: {d['best_params']['drop_out']}")
                st.write(f"**Best accuracy**: {d['best_accuracy'] * 100} %")
                st.write(f"**Runtime**: {d['run_time']} s")

else:
    st.write("Page does not exist")

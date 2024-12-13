import pandas as pd
import pyddm as ddm

def create_samples_from_csv(file_path):
    df_rt = pd.read_csv(file_path)

    # Convert latency to seconds
    df_rt['Latency'] /= 1000

    # Remove short and long trials
    # df_rt = df_rt[df_rt["Latency"] > .1] # Remove trials less than 100ms
    df_rt = df_rt[df_rt["Latency"] < 2.5] # Remove trials greater than 1650ms

    # Remove catch trials
    mask = (df_rt['Score'] == 'Correct') | (df_rt['Score'] == 'Incorrect')
    df_rt = df_rt[mask]

    # Change correct and incorrect to 1 and 0
    df_rt = df_rt.replace('Correct', 1)
    df_rt = df_rt.replace('Incorrect', 0)

    # Change values into correct values
    df_rt['AMRate'].replace(10, 9.77, inplace=True)
    df_rt['AMRate'].replace(8, 7.82, inplace=True)

    # Add "left_is_correct" column:
    left_corr_vals = (df_rt['AMRate'] == 4) | (df_rt['AMRate'] == 5)
    left_corr_vals.replace(True,1, inplace=True)
    left_corr_vals.replace(False,0, inplace=True)
    df_rt.insert(column='left_is_correct', value=left_corr_vals, loc=len(df_rt.columns))

    # Separate into trial types
    df_rt_av = df_rt[df_rt["TrialType"] == 'Av']
    df_rt_a = df_rt[df_rt["TrialType"] == 'Aud']
    df_rt_v = df_rt[df_rt["TrialType"] == 'Vis']

    
    # Create samples
    av_samp = ddm.Sample.from_pandas_dataframe(df_rt_av, rt_column_name="Latency", choice_column_name="Score")
    a_samp = ddm.Sample.from_pandas_dataframe(df_rt_a, rt_column_name="Latency", choice_column_name="Score")
    v_samp = ddm.Sample.from_pandas_dataframe(df_rt_v, rt_column_name="Latency", choice_column_name="Score")

    return av_samp, a_samp, v_samp

def get_amrates_from_csv(file_path):
    df_rt = pd.read_csv(file_path)

    # Change values into correct values
    df_rt['AMRate'].replace(10, 9.77, inplace=True)
    df_rt['AMRate'].replace(8, 7.82, inplace=True)

    df_rt_av = df_rt[df_rt["TrialType"] == 'Av']
    df_rt_a = df_rt[df_rt["TrialType"] == 'Aud']
    df_rt_v = df_rt[df_rt["TrialType"] == 'Vis']

    return df_rt_av, df_rt_a, df_rt_v
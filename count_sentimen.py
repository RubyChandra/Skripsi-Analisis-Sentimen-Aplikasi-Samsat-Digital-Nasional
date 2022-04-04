import pandas as pd

def countEachSentiment(df):
	# ambil jumlah positif
    df_pos = df.loc[df['label']==1]
    # ambil jumlah negatif
    df_neg = df.loc[df['label']==-1]
    # ambil jumlah neutral
    df_net = df.loc[df['label']==0]
    jlh= len(df_pos.index),len(df_neg.index),len(df_net.index)
    return jlh

def separateSentiment(df,x):
    df = df.loc[df['label']==x]
    return df
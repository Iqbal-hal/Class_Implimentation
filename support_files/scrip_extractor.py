# (No functional changes expected here)
# If this file imports via the package root, update it:
# from TradingWorkbench.support_files.File_IO import something


def scrip_extractor(master_df):
    '''
    master_df contains multiple scrips which is concateneated already.The function extract each scrips dataframe from the master_df.

    '''
    master_scrips_list = master_df['Stock'].unique()
    for scrip in master_scrips_list:
        scrip_df=master_df.loc[master_df['Stock']==scrip].copy()
        yield scrip,scrip_df

def scripdf_extractor(master_df):
    '''
    master_df contains multiple scrips which is concateneated already.The function extract each scrips dataframe from the master_df.

    '''
    master_scrips_list = master_df['Stock'].unique()
    for scrip in master_scrips_list:
        scrip_df=master_df.loc[master_df['Stock']==scrip].copy()
        yield scrip_df

    

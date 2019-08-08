get_end_tidal(df):
# get the troughs of the O2 data
        low_O2, _ = sg.find_peaks(df.O2.apply(lambda x:x*-1), prominence=prom)

        # create scatterplot of all O2 data
        if verb:
            print("Creating O2 plot ", file)
        sns.lineplot(x='Time', y='O2', data=df, linewidth=1, color='b')

        # get the data points of peak
        O2_df = df.iloc[low_O2]

        # linear interpolate the number of data points to match the scan Time
        O2_fxn = interpolate.interp1d(O2_df.Time, O2_df.O2, fill_value='extrapolate')
        et_O2 = O2_fxn(df.Time)

        # add peak overlay onto the scatterplot
        sns.lineplot(x=df.Time, y=et_O2, linewidth=2, color='g')
        plt.show()
        plt.close()

        # ask user if the peak finding was good
        ans = input("Was the output good enough (y/n)? \nNote: anything not starting with 'y' is considered 'n'.\n")
        bad = True if ans == '' or ans[0].lower() != 'y' else False
        if bad:
            print("The following variables can be changed: ")
            print("    1. prominence - Required prominence of peaks. Type: int")
            try:
                prom = int(input("New prominence (Default is 1): "))
            except:
                print("Default value used")
                prom = 1


    # make another loop for user confirmation that CO2 peak detection is good
    bad = True
    prom = 1
    while bad:
        # get peaks of the CO2 data
        high_CO2, _ = sg.find_peaks(df.CO2, prominence=prom)

        # create scatter of all CO2 data
        if verb:
            print('Creating CO2 plot ', file)
        sns.lineplot(x='Time', y='CO2', data=df, linewidth=1, color='b')

        # get the data points of peak
        CO2_df = df.iloc[high_CO2]

        # linear interpolate the number of data points to match the scan Time
        CO2_fxn = interpolate.interp1d(CO2_df.Time, CO2_df.CO2, fill_value='extrapolate')
        et_CO2 = CO2_fxn(df.Time)

        # add peak overlay onto the scatterplot
        sns.lineplot(x=df.Time, y=et_CO2, linewidth=2, color='r')
        plt.show()
        plt.close()

        # ask user if the peak finding was good
        ans = input("Was the output good enough (y/n)? \nNote: anything not starting with 'y' is considered 'n'.\n")
        bad = True if ans == '' or ans[0].lower() != 'y' else False
        if bad:
            print("The following variables can be changed: ")
            print("    1. prominence - Required prominence of peaks. Type: int")
            try:
                prom = int(input("New prominence (Default is 1): "))
            except:
                print("Default value used")
                prom = 1

    plt.close()

    # create subplots for png file later
    f, axes = plt.subplots(2, 1)

    # recreate the plot because plt.show clears plt
    sns.lineplot(x='Time', y='O2', data=df, linewidth=1, color='b', ax=axes[0])
    sns.lineplot(x=df.Time, y=et_O2, linewidth=2, color='g', ax=axes[0])
    # save the plot
    if verb:
        print('Saving plots for', file)
    # recreate the plot because plt.show clears plt
    sns.lineplot(x='Time', y='CO2', data=df, linewidth=1, color='b', ax=axes[1])
    sns.lineplot(x=df.Time, y=et_CO2, linewidth=2, color='r', ax=axes[1])

    save_path = save_path = path+file[:len(file)-4]+'/graph.png'
    f.savefig(save_path)
    if verb:
        print('Saving complete')
    f.clf()

    # since the find_peaks only returns the index that the peak is found, we need to grab the actual data point
    O2_df = df.iloc[low_O2].O2
    CO2_df = df.iloc[high_CO2].CO2

    # make the file name and save the O2 and CO2 data into their corresponding file
    if verb:
        print("Saving O2 data for ", file)
    save_path = path+file[:len(file)-4]+'/O2_contrast.txt'
    np.savetxt(save_path, et_O2, delimiter='\t')

    if verb:
        print('Saving CO2 data for ', file)
    save_path = path+file[:len(file)-4]+'/CO2_contrast.txt'
    np.savetxt(save_path, et_CO2, delimiter='\t')

    if verb:
        print()


    return et_O2, et_CO2

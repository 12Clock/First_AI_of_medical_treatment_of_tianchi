from save_result import save_results
import pickle as pkl

if __name__ == '__main__':

    list_file = open('./val_prediction_list.pkl', 'rb')
    position = pkl.load(list_file)


    reaults = save_results(csvfile_path='./test_csv/annotations.csv')
    reaults.write(position)
    list_file.close()
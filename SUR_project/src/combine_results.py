import argparse

#normalizacia dat pre GMM
def NormalizeData(data,minimum,maximum):
    return (data - minimum) / (maximum - minimum)

#vyhodnocovanie vysledkov
def get_faces_txt_and_evaluate(txt_for_face,dictionary,minimum,maximum):
    with open(txt_for_face) as file:
        lines = file.readlines()

    result = []
    for i in lines:
        i=i.strip()
        index_of_line = i.split(" ", 2)
        probability = dictionary[str(index_of_line[0])]
        if float(probability)>0 and float(index_of_line[1])>0.5:
            result.append((index_of_line[0],str(float(probability)+float(index_of_line[1])),1))
        elif float(probability)<0 and float(index_of_line[1])<0.5:
            result.append((index_of_line[0],str(abs(float(probability))+float(index_of_line[1])),0))
        elif float(index_of_line[1])>0.9:
            result.append((index_of_line[0],str(index_of_line[1]),1))
        elif(float(probability)>maximum/100*80):
            result.append((index_of_line[0],str(index_of_line[1]),1))
        elif float(index_of_line[1])>0.75:
            result.append((index_of_line[0],str(index_of_line[1]),1))
        elif (float(probability)>maximum/100*50):
            result.append((index_of_line[0],str(NormalizeData(float(probability),minimum,maximum)),1))
        elif float(index_of_line[1])>0.60:
            result.append((index_of_line[0],str(index_of_line[1]),1))
        elif(float(probability)>0):
            result.append((index_of_line[0],str(NormalizeData(float(probability),minimum,maximum)),1))
        else:
            result.append((index_of_line[0],str(index_of_line[1]),0))
    return result

#reprezentacie vysledkov z GMM vo forme dictionary
def get_dictionary_from_audio(txt_for_audio):
    with open(txt_for_audio) as file:
        lines = file.readlines()
    minimum=1
    maximum=-1

    result = dict()
    for i in lines:
        i=i.strip()
        index_of_line = i.split(" ", 2)
        result[str(index_of_line[0])]=index_of_line[1]
        if float(index_of_line[1]) > maximum:
            maximum = float(index_of_line[1])
        if float(index_of_line[1]) < minimum:
            minimum = float(index_of_line[1])
    return result,maximum,minimum

#vytvorenie finalneho prepisu
def write_result(save_results_dir,res):
    with open(save_results_dir, 'w') as f:
        for i in res:
            f.write(i[0] + ' ' + str(i[1]) + ' ' + str(i[2]) + '\n')


#parsovanie vstupnych argumentov
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt-for-face', help='')
    parser.add_argument('--txt-for-audio', help='')
    parser.add_argument('--save-results-dir', help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    if args.txt_for_face is not None:
        txt_for_face = args.txt_for_face
    if args.txt_for_audio is not None:
        txt_for_audio = args.txt_for_audio
    if args.save_results_dir is not None:
        save_results_dir = args.save_results_dir

    print("Loaded txt files")
    result,maximum,minimum = get_dictionary_from_audio(txt_for_audio)
    print("Making decisions based on score")
    res = get_faces_txt_and_evaluate(txt_for_face,result,minimum,maximum)
    print("Writing results")
    write_result(save_results_dir,res)

if __name__ == "__main__":
    main()


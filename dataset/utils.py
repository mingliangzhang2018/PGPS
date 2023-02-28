punctuation_list = ['.', '?', ',']
digit_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
capital_letter_list = [chr(item) for item in range(65, 91)]
low_letter_list = [chr(item) for item in range(97, 123)]
begin_words = ["find", "what", "solve", "determine", "express", "how"]
end_words = [".", ",", '?', "if", "so", "for which", "given", "with", "on",
             "in", "must", 'for', 'that', 'formed']
unit_list = ["mm^{2}", "cm^{2}", "in^{2}", "ft^{2}",
             "yd^{2}", "km^{2}", "units^{2}", "mi^{2}", "m^{2}"]
special_token_list = ['\\frac', '\\pi', '\\sqrt', "+", "-", "^"]


def get_token(ss): 
    """
        Tokenizer: divide the textual problem into words
    """
    raw_str_list = ss.strip().split(' ')
    # Split punctuation
    new_str1_list = []
    for item in raw_str_list:
        if item[-1] in punctuation_list:
            new_str1_list.append(item[:-1])
            new_str1_list.append(item[-1])
        else:
            new_str1_list.append(item)
    # Split points (capital letters)
    new_str2_list = []
    for item in new_str1_list:
        is_geo_rep = True
        point_list = []
        for k in item:
            if (ord(k) >= 65 and ord(k) <= 90) or \
                    ((k == '\'' or k in digit_list) and len(point_list) > 0):
                if k == '\'' or k in digit_list:
                    point_list[-1] += k
                else:
                    point_list.append(k)
            else:
                is_geo_rep = False
                break
        if is_geo_rep:
            new_str2_list += point_list
        else:
            new_str2_list.append(item.lower())

    return new_str2_list

def split_text(text_data):
    """
        split textual problem into condition and problem(target)
    """
    if len(text_data.token) == 0:
        return
    begin_ind = 0
    end_ind = len(text_data.token)
    for id, token in enumerate(text_data.token):
        if token in begin_words:
            begin_ind = id
            break
    for id in range(begin_ind+2, len(text_data.token)):
        if text_data.token[id] in end_words:
            if text_data.token[id] in punctuation_list:
                end_ind = id + 1
            else:
                end_ind = id
            break
    text_data.sect_tag = ['[COND]']*len(text_data.token[:begin_ind]) + \
                            ['[PROB]']*len(text_data.token[begin_ind: end_ind]) + \
                            ['[COND]']*len(text_data.token[end_ind:])

def get_point_angleID_tag(text_data, stru_data, sem_data):
    for id, item in enumerate(text_data.token):
        if item[0] in capital_letter_list:
           text_data.class_tag[id] = '[POINT]'
        if item.isdigit() and id > 0 and text_data.token[id-1] == "\\angle":
           text_data.class_tag[id] = '[ANGID]'

    for k in range(len(stru_data.token)):
        for id, item in enumerate(stru_data.token[k]):
            if item[0] in capital_letter_list:
                stru_data.class_tag[k][id] = '[POINT]'
            if item.isdigit() and id > 0 and stru_data.token[k][id-1] == "\\angle":
                stru_data.class_tag[k][id] = '[ANGID]'

    for k in range(len(sem_data.token)):
        for id, item in enumerate(sem_data.token[k]):
            if item[0] in capital_letter_list:
                sem_data.class_tag[k][id] = '[POINT]'
            if item.isdigit() and id > 0 and sem_data.token[k][id-1] == "\\angle":
                sem_data.class_tag[k][id] = '[ANGID]'

def get_args(token):
    letter_list = []
    for special_token in special_token_list:
        token = token.replace(special_token, "")
    for letter in token:
        if letter in low_letter_list and not letter in letter_list:
            letter_list.append(letter)
    return letter_list

def get_num_arg_tag(text_data, sem_data):
    """
        Determine the variables/arguments in the text condition
    """
    arg_sem_flat = []
    for k in range(len(sem_data.token)):
        if len(sem_data.token[k]) >= 3 and sem_data.token[k][-3] == '=':
            sem_data.class_tag[k][-2] = '[NUM]'
            arg_sem_flat += get_args(sem_data.token[k][-2])

    for id, token in enumerate(text_data.token):
        if text_data.sect_tag[id] == '[COND]' and text_data.class_tag[id] == '[GEN]':
            # unit symbol
            if token in unit_list:
                continue
            # digit existing (rough judgment)
            for word in digit_list:
                if word in token:
                    text_data.class_tag[id] = '[NUM]'
                    break
            # There are special characters, but not only special characters
            for word in special_token_list:
                if word in token and word != token:
                    text_data.class_tag[id] = '[NUM]'
                    break
            # Single lowercase letter, but not special cases
            if text_data.token[id] in low_letter_list:
                if id < len(text_data.token)-1 and text_data.token[id+1] == '=':
                    continue
                if text_data.token[id] == 'm' and id < len(text_data.token)-1 and text_data.token[id+1] in ["\\angle", "\\widehat"]:
                    continue
                if text_data.token[id] == 'a' and (id == 0 or text_data.token[id-1] != '='):
                    continue
                if not text_data.token[id] in arg_sem_flat and \
                    id > 0 and ('line' in text_data.token[id-1] or text_data.token[id-1] == 'and' or
                                (text_data.token[id-1] == ',' and text_data.token[id+1] == ',')):
                    continue
                text_data.class_tag[id] = '[NUM]'

    arg_text_flat = []
    for id, token in enumerate(text_data.token):
        if text_data.sect_tag[id] == '[COND]' and text_data.class_tag[id] == '[NUM]':
            arg_text_flat += get_args(token)

    # Determine arguments
    arg_all_flat = arg_text_flat + arg_sem_flat
    for id, token in enumerate(text_data.token):
        if text_data.class_tag[id] == '[GEN]' \
                and text_data.token[id] in arg_all_flat:
            if id < len(text_data.token)-1 and text_data.token[id+1] == '=':
                text_data.class_tag[id] = '[ARG]'
                continue
            if text_data.token[id] == 'm' and id < len(text_data.token)-1 and text_data.token[id+1] in ["\\angle", "\\widehat"]:
                continue
            if text_data.token[id] == 'a' and (id == 0 or text_data.token[id-1] != '=') and \
                                text_data.sect_tag[id]=='[COND]':
                continue
            if id > 0 and ('line' in text_data.token[id-1] or text_data.token[id-1] == 'and' or
                           (text_data.token[id-1] == ',' and text_data.token[id+1] == ',')):
                continue
            text_data.class_tag[id] = '[ARG]'

def remove_sem_dup(text_data, sem_data, exp_token):
    """
        Remove the seq of sem_data if num is also in the text_data
        and change the corresponding expression
    """
    text_num_list, id_all_list, id_map_list = [], [], []
    token_, sect_tag_, class_tag_ = [], [], []

    for k in range(len(text_data.token)):
        if text_data.class_tag[k] == '[NUM]':
            text_num_list.append(text_data.token[k])
            var_name = 'N'+str(len(id_all_list))
            id_all_list.append(var_name)
            id_map_list.append(var_name)

    for k in range(len(sem_data.token)):
        if sem_data.class_tag[k][-2] == '[NUM]':
            var_name = 'N'+str(len(id_all_list))
            id_all_list.append(var_name)  
            if not sem_data.token[k][-2] in text_num_list:
                token_.append(sem_data.token[k])
                sect_tag_.append(sem_data.sect_tag[k])
                class_tag_.append(sem_data.class_tag[k])
                id_map_list.append(var_name)
        else:
            token_.append(sem_data.token[k])
            sect_tag_.append(sem_data.sect_tag[k])
            class_tag_.append(sem_data.class_tag[k])

    num_map_dict = {key:value for key, value in zip(id_map_list, id_all_list)} 
    for k in range(len(exp_token)):
        if exp_token[k] in num_map_dict:
            exp_token[k] = num_map_dict[exp_token[k]]

    sem_data.token = token_
    sem_data.sect_tag = sect_tag_
    sem_data.class_tag = class_tag_
    
def get_combined_text(text_seq, stru_seqs, sem_seqs, combine_text, args):
    '''
        combination style: [stru_seqs, text_cond, sem_seqs, text_prob]
    '''
    # split cond and prob in text_seq
    begin_ind = end_ind = None
    for k in range(len(text_seq.sect_tag)):
        if text_seq.sect_tag[k]=='[PROB]': 
            begin_ind = k
            break
    for k in range(len(text_seq.sect_tag)-1,-1,-1):
        if text_seq.sect_tag[k]=='[PROB]':
            end_ind = k+1
            break
    # combine text_seq, stru_seqs and sem_seqs
    for key in vars(combine_text):
        # get text_cond and text_prob
        text_all_value = getattr(text_seq, key)
        text_cond_value = text_all_value[:begin_ind] + text_all_value[end_ind:]
        text_prob_value = text_all_value[begin_ind:end_ind]
        if args.without_stru:
            value_all = text_cond_value + sum(getattr(sem_seqs, key), []) + text_prob_value
        else:
            value_all = sum(getattr(stru_seqs, key), []) + text_cond_value + \
                                sum(getattr(sem_seqs, key), []) + text_prob_value
                
        setattr(combine_text, key, value_all)
    
def get_var_arg(combine_text, args):

    var_values, arg_values = [], []
    var_positions, arg_positions = [], []
    class_tag = combine_text.class_tag
    token = combine_text.token

    for k in range(len(class_tag)):
        if class_tag[k] == '[NUM]':
            var_values.append(token[k])
            var_positions.append(k)
        if class_tag[k] == '[ARG]':
            arg_values.append(token[k])
            arg_positions.append(k)
    # merge position of var and arg
    return  var_positions+arg_positions, var_values, arg_values

def get_text_index(combine_text, src_lang):
 
    text_sect_tag = src_lang.indexes_from_sentence(combine_text.sect_tag, id_type='sect_tag')
    text_class_tag = src_lang.indexes_from_sentence(combine_text.class_tag, id_type='class_tag')
    text_token = [combine_text.token[:], ['[PAD]']*len(combine_text.token)]

    for k in range(len(combine_text.class_tag)):
        if combine_text.class_tag[k] == '[NUM]':
            letter_list = get_args(combine_text.token[k])
            text_token[0][k] = text_token[1][k] = "[PAD]"
            for j in range(len(letter_list)):
                text_token[j][k] = letter_list[j]
    text_token = [src_lang.indexes_from_sentence(item, id_type='text') for item in text_token]
    
    return text_token, text_sect_tag, text_class_tag


            



    
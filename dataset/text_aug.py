import random

upper_case_list = [chr(i) for i in range(65, 91)]
low_case_list = [chr(i) for i in range(97, 123)]
angle_id_list = [str(i) for i in range(1, 21)]
spec_token_list = ['frac', 'pi', 'sqrt']

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, text_seq, stru_seqs, sem_seqs, exp):
        for t in self.transforms:
            t(text_seq, stru_seqs, sem_seqs, exp)

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

class Point_RandomReplace(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def get_point_map(self):
        value_list = [chr(i) for i in range(65, 91)]
        random.shuffle(value_list)
        map_dict = {key:value for key, value in zip(upper_case_list, value_list)} 
        return map_dict        

    def __call__(self, text_seq, stru_seqs, sem_seqs, exp):
        if random.random() < self.prob:
            map_dict = self.get_point_map()
            for k in range(len(text_seq.token)):
                if text_seq.class_tag[k] == '[POINT]':
                    text_seq.token[k] = map_dict[text_seq.token[k][0]]
            for k in range(len(stru_seqs.token)):
                for j in range(len(stru_seqs.token[k])):
                    if stru_seqs.class_tag[k][j] == '[POINT]':
                        stru_seqs.token[k][j] = map_dict[stru_seqs.token[k][j][0]]
            for k in range(len(sem_seqs.token)):
                for j in range(len(sem_seqs.token[k])):
                    if sem_seqs.class_tag[k][j] == '[POINT]':
                        sem_seqs.token[k][j] = map_dict[sem_seqs.token[k][j][0]]

class AngID_RandomReplace(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def get_angid_map(self):
        value_list = [str(i) for i in range(1, 21)]
        random.shuffle(value_list)
        map_dict = {key:value for key, value in zip(angle_id_list, value_list)} 
        return map_dict        

    def __call__(self, text_seq, stru_seqs, sem_seqs, exp):
        if random.random() < self.prob:
            map_dict = self.get_angid_map()
            for k in range(len(text_seq.token)):
                if text_seq.class_tag[k] == '[ANGID]':
                    text_seq.token[k] = map_dict[text_seq.token[k]]
            for k in range(len(sem_seqs.token)):
                for j in range(len(sem_seqs.token[k])):
                    if sem_seqs.class_tag[k][j] == '[ANGID]':
                        sem_seqs.token[k][j] = map_dict[sem_seqs.token[k][j]]

class Arg_RandomReplace(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def get_arg_map(self):
        value_list = [chr(i) for i in range(97, 123)]
        random.shuffle(value_list)
        map_dict = {key:value for key, value in zip(low_case_list, value_list)} 
        return map_dict
    
    def map_arg_in_num(self, map_dict, num):
        num_t = num[:]
        new_num = ''
        for item in spec_token_list:
            num_t = num_t.replace(item, "@"*len(item))
        for k in range(len(num_t)):
            if num_t[k]!='@' and num[k] in low_case_list:
                new_num += map_dict[num[k]]
            else:
                new_num += num[k]
        return new_num

    def __call__(self, text_seq, stru_seqs, sem_seqs, exp):
        if random.random() < self.prob:
            map_dict = self.get_arg_map()
            for k in range(len(text_seq.token)): 
                if text_seq.class_tag[k] == '[NUM]':
                    text_seq.token[k] = self.map_arg_in_num(map_dict, text_seq.token[k])
                if text_seq.class_tag[k] == '[ARG]':
                    text_seq.token[k] = map_dict[text_seq.token[k]]
            for k in range(len(sem_seqs.token)):
                for j in range(len(sem_seqs.token[k])):
                    if sem_seqs.class_tag[k][j] == '[NUM]':
                        sem_seqs.token[k][j] = self.map_arg_in_num(map_dict, sem_seqs.token[k][j])
            for k in range(len(exp)):
                if exp[k] in low_case_list:
                    exp[k] = map_dict[exp[k]]

class StruPoint_RandomRotate(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def get_seq_points(self, class_tag):
        id_list = []
        begin_point_id = end_point_id = None
        for id, token in enumerate(class_tag):
            if token == '[POINT]':
                if begin_point_id is None:
                    begin_point_id = id
            elif not begin_point_id is None and end_point_id is None:
                end_point_id = id 
                id_list.append([begin_point_id, end_point_id])
                begin_point_id = end_point_id = None
        if not begin_point_id is None and end_point_id is None:
            id_list.append([begin_point_id, len(class_tag)])
        
        return id_list[-1][0], id_list[-1][1]

    def __call__(self, text_seq, stru_seqs, sem_seqs, exp):
        for k in range(len(stru_seqs.token)):
            if random.random() < self.prob:
                begin_id, end_id = self.get_seq_points(stru_seqs.class_tag[k])
                # point on line
                if stru_seqs.token[k][0] == 'line':
                    stru_seqs.token[k][begin_id:end_id] = stru_seqs.token[k][end_id-1:begin_id-1:-1]
                # point on circle
                if stru_seqs.token[k][0] == '\\odot':
                    # clockwise change 
                    if random.random() < 0.5: 
                        stru_seqs.token[k][begin_id:end_id] = stru_seqs.token[k][end_id-1:begin_id-1:-1]
                    # set initial point
                    init_loc = random.randint(begin_id, end_id-1) 
                    stru_seqs.token[k][begin_id:end_id] = stru_seqs.token[k][init_loc:end_id] + \
                                                                stru_seqs.token[k][begin_id:init_loc]

class SemPoint_RandomRotate(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def get_seq_points(self, class_tag):
        id_list = []
        begin_point_id = end_point_id = None
        for id, token in enumerate(class_tag):
            if token == '[POINT]':
                if begin_point_id is None:
                    begin_point_id = id
            elif not begin_point_id is None and end_point_id is None:
                end_point_id = id 
                id_list.append((begin_point_id, end_point_id-1))
                begin_point_id = end_point_id = None
        if not begin_point_id is None and end_point_id is None:
            id_list.append((begin_point_id, len(class_tag)-1))
        
        return id_list

    def __call__(self, text_seq, stru_seqs, sem_seqs, exp):
        if random.random() < self.prob:
            for k in range(len(sem_seqs.token)):
                id_list = self.get_seq_points(sem_seqs.class_tag[k])
                for begin_id, end_id in id_list:
                    if random.random() < self.prob:  
                        sem_seqs.token[k][begin_id], sem_seqs.token[k][end_id] = \
                            sem_seqs.token[k][end_id], sem_seqs.token[k][begin_id]                   

class SemSeq_RandomRotate(object):

    def __init__(self, prob=0.5):
        self.prob = prob + 0.2

    def __call__(self, text_seq, stru_seqs, sem_seqs, exp):
        if random.random() < self.prob:
            # varible id 
            num_all_list, num_sem_list, num_map_list = [], [], []
            for item in text_seq.class_tag:
                if item=='[NUM]':
                    var_name = 'N'+str(len(num_all_list))
                    num_all_list.append(var_name)
                    num_map_list.append(var_name)
            for k in range(len(sem_seqs.token)):
                if sem_seqs.class_tag[k][-2] == '[NUM]':
                    var_name = 'N'+str(len(num_all_list))
                    num_all_list.append(var_name)
                    num_sem_list.append([var_name])
                else:
                    num_sem_list.append([])
            # shuffle sem_seq
            if len(sem_seqs.token)>0:
                random_id_list = [k for k in range(len(sem_seqs.token))] 
                random.shuffle(random_id_list)
                for key,value in vars(sem_seqs).items():
                    _, value = zip(*sorted(zip(random_id_list, value)))
                    setattr(sem_seqs, key, list(value))
                _, num_sem_list = zip(*sorted(zip(random_id_list, num_sem_list)))
            # expression map 
            for k in range(len(sem_seqs.token)):
                num_map_list += num_sem_list[k]
            num_map_dict = {key:value for key, value in zip(num_map_list, num_all_list)} 
            for k in range(len(exp)):
                if exp[k] in num_map_dict:
                    exp[k] = num_map_dict[exp[k]]

class StruSeq_RandomRotate(object):

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, text_seq, stru_seqs, sem_seqs, exp):
        if random.random() < self.prob:
            # shuffle stru_seq
            if len(stru_seqs.token)>0:
                random_id_list = [k for k in range(len(stru_seqs.token))] 
                random.shuffle(random_id_list)
                for key, value in vars(stru_seqs).items():
                    _, value = zip(*sorted(zip(random_id_list, value)))
                    setattr(stru_seqs, key, list(value))

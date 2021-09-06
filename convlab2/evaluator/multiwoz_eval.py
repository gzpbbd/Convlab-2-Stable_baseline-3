# -*- coding: utf-8 -*-

import re
import numpy as np
from copy import deepcopy
from pprint import pprint
from convlab2.evaluator.evaluator import Evaluator
from convlab2.util.multiwoz.dbquery import Database

requestable = \
    {'attraction': ['post', 'phone', 'addr', 'fee', 'area', 'type'],
     'restaurant': ['addr', 'phone', 'post', 'ref', 'price', 'area', 'food'],
     'train': ['ticket', 'time', 'ref', 'id', 'arrive', 'leave'],
     'hotel': ['addr', 'post', 'phone', 'ref', 'price', 'internet', 'parking', 'area', 'type', 'stars'],
     'taxi': ['car', 'phone'],
     'hospital': ['post', 'phone', 'addr'],
     'police': ['addr', 'post', 'phone']}

belief_domains = requestable.keys()

mapping = {'restaurant': {'addr': 'address', 'area': 'area', 'food': 'food', 'name': 'name', 'phone': 'phone',
                          'post': 'postcode', 'price': 'pricerange'},
           'hotel': {'addr': 'address', 'area': 'area', 'internet': 'internet', 'parking': 'parking', 'name': 'name',
                     'phone': 'phone', 'post': 'postcode', 'price': 'pricerange', 'stars': 'stars', 'type': 'type'},
           'attraction': {'addr': 'address', 'area': 'area', 'fee': 'entrance fee', 'name': 'name', 'phone': 'phone',
                          'post': 'postcode', 'type': 'type'},
           'train': {'id': 'trainID', 'arrive': 'arriveBy', 'day': 'day', 'depart': 'departure', 'dest': 'destination',
                     'time': 'duration', 'leave': 'leaveAt', 'ticket': 'price'},
           'taxi': {'car': 'car type', 'phone': 'phone'},
           'hospital': {'post': 'postcode', 'phone': 'phone', 'addr': 'address', 'department': 'department'},
           'police': {'post': 'postcode', 'phone': 'phone', 'addr': 'address'}}

time_re = re.compile(r'^(([01]\d|2[0-4]):([0-5]\d)|24:00)$')
NUL_VALUE = ["", "dont care", 'not mentioned', "don't care", "dontcare", "do n't care"]


class MultiWozEvaluator(Evaluator):
    def __init__(self):
        self.sys_da_array = []
        self.usr_da_array = []
        self.goal = {}
        self.cur_domain = ''
        self.booked = {}
        self.database = Database()
        self.dbs = self.database.dbs

    def _init_dict(self):
        dic = {}
        for domain in belief_domains:
            dic[domain] = {'info': {}, 'book': {}, 'reqt': []}
        return dic

    def _init_dict_booked(self):
        dic = {}
        for domain in belief_domains:
            dic[domain] = None
        return dic

    def _expand(self, _goal):
        '''
        补全 _goal，使得每个 domain 都有 info, book 与 reqt 三项（用空字典填充）
        :param _goal:
        :return: 补全后的 goal
        '''
        goal = deepcopy(_goal)
        for domain in belief_domains:
            if domain not in goal:
                goal[domain] = {'info': {}, 'book': {}, 'reqt': []}
                continue
            if 'info' not in goal[domain]:
                goal[domain]['info'] = {}
            if 'book' not in goal[domain]:
                goal[domain]['book'] = {}
            if 'reqt' not in goal[domain]:
                goal[domain]['reqt'] = []
        return goal

    def add_goal(self, goal):
        """init goal and array

        args:
            goal:
                dict[domain] dict['info'/'book'/'reqt'] dict/dict/list[slot]
        """
        self.sys_da_array = []
        self.usr_da_array = []
        self.goal = goal
        self.cur_domain = ''
        self.booked = self._init_dict_booked()

    def add_sys_da(self, da_turn):
        '''
        有两个作用：
        (1) 将 system action 保存到列表 self.sys_da_array 中。每个 system action 转为字符串'domain-intent-slot-value'
        (2) action 的 value 是 database 中实体的编号。根据 system action，将预定的 实体信息填入 self.booked 中。

        :param da_turn: system action 组成的列表，即 list[intent, domain, slot, value]
        :return:
        '''
        for intent, domain, slot, value in da_turn:
            dom_int = '-'.join([domain, intent])
            domain = dom_int.split('-')[0].lower()
            if domain in belief_domains and domain != self.cur_domain:
                self.cur_domain = domain
            da = (dom_int + '-' + slot).lower()
            value = str(value)
            self.sys_da_array.append(da + '-' + value)

            if da == 'booking-book-ref' and self.cur_domain in ['hotel', 'restaurant', 'train']:
                if not self.booked[self.cur_domain] and re.match(r'^\d{8}$', value) and \
                        len(self.dbs[self.cur_domain]) > int(value):
                    self.booked[self.cur_domain] = self.dbs[self.cur_domain][int(value)].copy()
                    self.booked[self.cur_domain]['Ref'] = value
            elif da == 'train-offerbooked-ref' or da == 'train-inform-ref':
                if not self.booked['train'] and re.match(r'^\d{8}$', value) and len(self.dbs['train']) > int(value):
                    self.booked['train'] = self.dbs['train'][int(value)].copy()
                    self.booked['train']['Ref'] = value
            elif da == 'taxi-inform-car':
                if not self.booked['taxi']:
                    self.booked['taxi'] = 'booked'

    def add_usr_da(self, da_turn):
        '''
        将 user action 保存到列表 self.usr_da_array 中。每个 user action 转为字符串'domain-intent-slot-value'
        :param da_turn: user action 组成的列表
        :return:
        '''
        """add usr_da into array

        args:
            da_turn:
                list[intent, domain, slot, value]
        """
        for intent, domain, slot, value in da_turn:
            dom_int = '-'.join([domain, intent])
            domain = dom_int.split('-')[0].lower()
            if domain in belief_domains and domain != self.cur_domain:
                self.cur_domain = domain
            da = (dom_int + '-' + slot).lower()
            value = str(value)
            self.usr_da_array.append(da + '-' + value)

    def _book_rate_goal(self, goal, booked_entity, domains=None):
        '''
        对每个 domain 进行检查，判断选中的实体是否满足 goal。

        :param goal: 用户 goal
        :param booked_entity: 预定的实体
        :param domains: 检查哪些 domain？None 表示检查所有 domain。
        :return: 一个列表，列表中每个数字对应一个 domain 的 score. 1 为 符合要求，0 为不符合，(0,1)间的数为部分符合要求
        '''
        if domains is None:
            domains = belief_domains
        score = []
        for domain in domains:
            if 'book' in goal[domain] and goal[domain]['book']:
                tot = len(goal[domain]['info'].keys())
                if tot == 0:
                    continue
                entity = booked_entity[domain]
                if entity is None:
                    score.append(0)
                    continue
                if domain == 'taxi':
                    score.append(1)
                    continue
                match = 0
                for k, v in goal[domain]['info'].items():
                    if k in ['destination', 'departure']:
                        tot -= 1
                    elif k == 'leaveAt':
                        try:
                            v_constraint = int(v.split(':')[0]) * 100 + int(v.split(':')[1])
                            v_select = int(entity['leaveAt'].split(':')[0]) * 100 + int(entity['leaveAt'].split(':')[1])
                            if v_constraint <= v_select:
                                match += 1
                        except (ValueError, IndexError):
                            match += 1
                    elif k == 'arriveBy':
                        try:
                            v_constraint = int(v.split(':')[0]) * 100 + int(v.split(':')[1])
                            v_select = int(entity['arriveBy'].split(':')[0]) * 100 + int(
                                entity['arriveBy'].split(':')[1])
                            if v_constraint >= v_select:
                                match += 1
                        except (ValueError, IndexError):
                            match += 1
                    else:
                        if v.strip() == entity[k].strip():
                            match += 1
                if tot != 0:
                    score.append(match / tot)
        return score

    def _inform_F1_goal(self, goal, sys_history, domains=None):
        '''
        judge if all the requested information is answered

        :param goal: user goal
        :param sys_history: 对话中所有 system action 组成的列表
        :param domains:
        :return: TP, FP, FN, bad_inform, reqt_not_inform, inform_not_reqt
        '''
        if domains is None:
            domains = belief_domains
        inform_slot = {}
        for domain in domains:
            inform_slot[domain] = set()
        TP, FP, FN = 0, 0, 0

        inform_not_reqt = set()
        reqt_not_inform = set()
        bad_inform = set()

        for da in sys_history:
            domain, intent, slot, value = da.split('-', 3)
            if intent in ['inform', 'recommend', 'offerbook', 'offerbooked'] and \
                    domain in domains and slot in mapping[domain] and value.strip() not in NUL_VALUE:
                key = mapping[domain][slot]
                if self._check_value(domain, key, value):
                    # print('add key', key)
                    inform_slot[domain].add(key)
                else:
                    bad_inform.add((intent, domain, key))
                    FP += 1

        for domain in domains:
            for k in goal[domain]['reqt']:
                if k in inform_slot[domain]:
                    # print('k: ', k)
                    TP += 1
                else:
                    # print('FN + 1')
                    reqt_not_inform.add(('request', domain, k))
                    FN += 1
            for k in inform_slot[domain]:
                # exclude slots that are informed by users
                if k not in goal[domain]['reqt'] \
                        and k not in goal[domain]['info'] \
                        and k in requestable[domain]:
                    # print('FP + 1 @2', k)
                    inform_not_reqt.add(('inform', domain, k,))
                    FP += 1
        return TP, FP, FN, bad_inform, reqt_not_inform, inform_not_reqt

    def _check_value(self, domain, key, value):
        if key == "area":
            return value.lower() in ["centre", "east", "south", "west", "north"]
        elif key == "arriveBy" or key == "leaveAt":
            return time_re.match(value)
        elif key == "day":
            return value.lower() in ["monday", "tuesday", "wednesday", "thursday", "friday",
                                     "saturday", "sunday"]
        elif key == "duration":
            return 'minute' in value
        elif key == "internet" or key == "parking":
            return value in ["yes", "no", "none"]
        elif key == "phone":
            return re.match(r'^\d{11}$', value) or domain == "restaurant"
        elif key == "price":
            return 'pound' in value
        elif key == "pricerange":
            return value in ["cheap", "expensive", "moderate", "free"] or domain == "attraction"
        elif key == "postcode":
            return re.match(r'^cb\d{1,3}[a-z]{2,3}$', value) or value == 'pe296fl'
        elif key == "stars":
            return re.match(r'^\d$', value)
        elif key == "trainID":
            return re.match(r'^tr\d{4}$', value.lower())
        else:
            return True

    def book_rate(self, ref2goal=True, aggregate=True):
        '''
        检查 self.booked 中的 entities 是否满足 goal

        :param ref2goal: True 使用当前的 self.goal； False 重新初始化 self.goal.
        :param aggregate: True 返回所有 domain score 的平均值；False 返回一个 score 列表，每个值对应一个 domain 的分数。
        :return: score. None 或者 [0, 1] 间的数， 0 表示不满足 goal，1 表示满足 goal
        '''
        if ref2goal:
            goal = self._expand(self.goal)
        else:
            goal = self._init_dict()
            for domain in belief_domains:
                if domain in self.goal and 'book' in self.goal[domain]:
                    goal[domain]['book'] = self.goal[domain]['book']
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in mapping[d]:
                    goal[d]['info'][mapping[d][s]] = v
        score = self._book_rate_goal(goal, self.booked)
        if aggregate:
            return np.mean(score) if score else None
        else:
            return score

    def inform_F1(self, ref2goal=True, aggregate=True):
        '''
        根据 self.goal 与 self.sys_da_array，计算指标 (prec, rec, F) 或者 (TP, FP, FN)

        :param ref2goal: 是否使用当前的 self.goal
        :param aggregate: True 返回 prec, rec, F; False 返回 TP, FP, FN
        :return: 根据 aggregate 返回不同的结果
        '''
        if ref2goal:
            goal = self._expand(self.goal)
        else:
            goal = self._init_dict()
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in mapping[d]:
                    goal[d]['info'][mapping[d][s]] = v
                elif i == 'request':
                    goal[d]['reqt'].append(s)
        TP, FP, FN, _, _, _ = self._inform_F1_goal(goal, self.sys_da_array)
        if aggregate:
            try:
                rec = TP / (TP + FN)
            except ZeroDivisionError:
                return None, None, None
            try:
                prec = TP / (TP + FP)
                F1 = 2 * prec * rec / (prec + rec)
            except ZeroDivisionError:
                return 0, rec, 0
            return prec, rec, F1
        else:
            return [TP, FP, FN]

    def task_success(self, ref2goal=True):
        '''
        检查任务是否完成（1. booked entities 满足要求， 2. system inform 了所有 user request 的信息）

        :param ref2goal: 是否参考当前的 self.goal
        :return: 1: 任务完成；0: 任务失败
        '''
        book_sess = self.book_rate(ref2goal)
        inform_sess = self.inform_F1(ref2goal)
        goal_sess = self.final_goal_analyze()
        # book rate == 1 & inform recall == 1
        if ((book_sess == 1 and inform_sess[1] == 1) \
            or (book_sess == 1 and inform_sess[1] is None) \
            or (book_sess is None and inform_sess[1] == 1)) \
                and goal_sess == 1:
            return 1
        else:
            return 0

    def domain_reqt_inform_analyze(self, domain, ref2goal=True):
        '''
        分析该轮对话中 self.goal 中 request 的信息与 system inform 的信息

        :param domain: str. domain name.
        :param ref2goal:
        :return: TP, FP, FN, bad_inform, reqt_not_inform, inform_not_reqt
        '''
        if domain not in self.goal:
            return None

        if ref2goal:
            goal = {}
            goal[domain] = self._expand(self.goal)[domain]
        else:
            goal = {}
            goal[domain] = {'info': {}, 'book': {}, 'reqt': []}
            if 'book' in self.goal[domain]:
                goal[domain]['book'] = self.goal[domain]['book']
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if d != domain:
                    continue
                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in mapping[d]:
                    goal[d]['info'][mapping[d][s]] = v
                elif i == 'request':
                    goal[d]['reqt'].append(s)

        inform = self._inform_F1_goal(goal, self.sys_da_array, [domain])
        return inform

    def domain_success(self, domain, ref2goal=True):
        '''
        判断指定的 domain 是否预定成功且 request 到了需要的信息

        :param domain:
        :param ref2goal:
        :return:
        '''
        """
        judge if the domain (subtask) is successfully completed
        """
        if domain not in self.goal:
            return None

        if ref2goal:
            goal = {}
            goal[domain] = self._expand(self.goal)[domain]
        else:
            goal = {}
            goal[domain] = {'info': {}, 'book': {}, 'reqt': []}
            if 'book' in self.goal[domain]:
                goal[domain]['book'] = self.goal[domain]['book']
            for da in self.usr_da_array:
                d, i, s, v = da.split('-', 3)
                if d != domain:
                    continue
                if i in ['inform', 'recommend', 'offerbook', 'offerbooked'] and s in mapping[d]:
                    goal[d]['info'][mapping[d][s]] = v
                elif i == 'request':
                    goal[d]['reqt'].append(s)

        book_rate = self._book_rate_goal(goal, self.booked, [domain])
        book_rate = np.mean(book_rate) if book_rate else None

        inform = self._inform_F1_goal(goal, self.sys_da_array, [domain])
        try:
            inform_rec = inform[0] / (inform[0] + inform[2])
        except ZeroDivisionError:
            inform_rec = None

        if (book_rate == 1 and inform_rec == 1) \
                or (book_rate == 1 and inform_rec is None) \
                or (book_rate is None and inform_rec == 1):
            return 1
        else:
            return 0

    def _final_goal_analyze(self):
        '''
        对每个 domain，都判断是否 system booked 的 entities 是否满足 self.goal
        :return: match, mismatch。match：满足要求的 domain 数量；mismatch：不满足要求的 domain 数量。
        '''
        match = mismatch = 0
        for domain, dom_goal_dict in self.goal.items():
            constraints = []
            if 'reqt' in dom_goal_dict:
                reqt_constraints = list(dom_goal_dict['reqt'].items())  # list[slot, value]
                constraints += reqt_constraints
            else:
                reqt_constraints = []
            if 'info' in dom_goal_dict:
                info_constraints = list(dom_goal_dict['info'].items())  # list[slot, value]
                constraints += info_constraints
            else:
                info_constraints = []
            query_result = self.database.query(domain, info_constraints, soft_contraints=reqt_constraints)
            if not query_result:
                mismatch += 1  # database 中没有符合 self.goal 的 entity
                continue

            booked = self.booked[domain]
            if not self.goal[domain].get('book'):
                match += 1  # 该 domain 不需要 booking
            elif isinstance(booked, dict):
                ref = booked['Ref']
                # 判断 database 中符合 goal 的 entities 是否是 system booked 的 entities
                if any(found['Ref'] == ref for found in query_result):
                    match += 1
                else:
                    mismatch += 1
            else:
                match += 1
        return match, mismatch

    def final_goal_analyze(self):
        '''
        分析 system 预定的 entities 是否满足 self.goal。
        :return: 满足 self.goal 的 domain 的比例。即 matched_domain / all_domain
        '''
        """percentage of domains, in which the final goal satisfies the database constraints.
        If there is no dialog action, returns 1."""
        match, mismatch = self._final_goal_analyze()
        if match == mismatch == 0:
            return 1
        else:
            return match / (match + mismatch)

    def get_reward(self):
        if self.task_success():
            reward = 40
        elif self.cur_domain and self.domain_success(self.cur_domain):
            reward = 5
        else:
            reward = -1
        return reward

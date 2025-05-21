def search_BBcode(l, m):
    r1,r2,r3,s1,s2,s3 = 0,0,0,0,0,0
    if r1+r2+r3+s1+s2+s3 == 0:
        with open(f'good_log_{l}_{m}', 'w') as f:
            pass
    else:
        with open(f'good_log_{l}_{m}', 'a') as f:
            pass
    gap = start([gap_path, '-L', 'workplace','-q', '-b'])
    power = get_candidate_state(1)
    power = [str(i) for i in range(6)]
    terms = [(0, str(i)) for i in range(l)] + [(1, str(i)) for i in range(1, m)]
    n = 2 * l * m
    x = get_x(l, m)
    y = get_y(l, m)
    result = []
    # r1,r2,r3,s1,s2,s3 = 0, 2, 3, 1, 19, 3
    
    # 9-8 0, 3, 6, 7, 11, 14
    #9-10 3, 12, 14, 6, 10, 11
    # r1,r2,r3,s1,s2,s3 = 0, 5, 10, 5, 17, 19 12-10 
    # 2, 7, 15, 7, 9, 17 15-5
    found = set()
    def same(p1, p2):
        if p1[0]==p2[0] and eval(p1[1]) == eval(p2[1]):
            return True
        return False
    def mat(p):
        M = x if p[0]==0 else y
        return mp(M, eval(p[1])) 
    
    def valid(p):
        if eval(p)<0 or eval(p)>15:
            return True
        return False
    term_len = len(terms)
    count = 0
    iter_count = 0
    flag = False
    for i1 in range(r1, term_len-2):
        a1 = terms[i1]
        if valid(a1[1]):
            continue
        r2 = i1+1 if flag else r2
        for i2 in range(r2, term_len-1):
            a2 = terms[i2]
            if valid(a2[1]):
                continue
            if same(a1, a2):
                continue
            r3 = i2+1 if flag else r3
            for i3 in range(r3, term_len):
                a3 = terms[i3]
                if valid(a3[1]):
                    continue
                if same(a1, a3) or same(a2, a3):
                    continue
                A = mat(a1) + mat(a2) + mat(a3)
                # print(rank(A))
                if rank(A)>=l*m-2:
                    # print(PowStr(a1), PowStr(a2), PowStr(a3))
                    continue
                s1 = i1 if flag else s1
                for j1 in range(s1, term_len-2):
                    b1 = terms[j1]
                    if valid(b1[1]):
                        continue
                    s2 = j1+1 if flag else s2
                    for j2 in range(s2, term_len-1):
                        b2 = terms[j2]
                        if valid(b2[1]):
                            continue
                        if same(b1, b2):
                            continue
                        
                        s3 = j2+1 if flag else s3
                        print(s3)
                        for j3 in range(s3, term_len):
                            iter_count += 1
                            flag = True
                            b3 = terms[j3]
                            if valid(b3[1]):
                                continue
                            if same(b1, b3) or same(b2, b3):
                                continue
                            # print(f"{i1}, {i2}, {i3}, {j1}, {j2}, {j3}")
                            # print((a1,a2,a3,b1,b2,b3))
                            B = mat(b1) + mat(b2) + mat(b3)
                            k, d = Get_kd_BBCode(gap, A, B, l, m)
                            if k!=0:
                                count+=1
                            if count>120:
                                terminate(gap)
                                gap = start([gap_path, '-L', 'workplace','-q', '-b'])      
                                count=0          
                            r_frac = k*1.0/(2.0*n) 
                                               
                            # print(f"good with n: {n}, k: {k}, d: {d}, r: {r_frac}")
                            # print(f"{iter_count}, {PowStr(a1)}+{PowStr(a2)}+{PowStr(a3)}, {PowStr(b1)}+{PowStr(b2)}+{PowStr(b3)}")
                            if True and good(n,k,d):
                                r = 2.0*n / k
                                found.add((n,k,d))
                                with open(f'good_log_{l}_{m}', 'a') as f:
                                    f.write((f"good with n: {n}, k: {k}, d: {d}, r: {r} "))
                                    f.write(f"{PowStr(a1)}+{PowStr(a2)}+{PowStr(a3)}, {PowStr(b1)}+{PowStr(b2)}+{PowStr(b3)}\n")
                                result.append((n,k,d,a1,a2,a3,b1,b2,b3))
    terminate(gap)
    sortFile(f'good_log_{l}_{m}')
    print(f"itercount: {iter_count}")
    if len(result)>0:
        os.system(f'rm {l}{m}Hx.mtx')
        os.system(f'rm {l}{m}Hz.mtx')
    return result
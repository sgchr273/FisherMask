import torch
import pickle
import os

def save_model(rd, net, filename, alg):
    if not os.path.exists("./Save/Models/" + filename):
        os.makedirs("./Save/Models/"+filename)
    torch.save(net.state_dict(), "./Save/Models/"+ filename + f"/{alg}_model_" +  str(rd)+ ".pt")

def load_model(rd, net, filename, alg):
    try:
        net.load_state_dict(torch.load("./Save/Models/"+ filename + f"/{alg}_model_" +  str(rd) + ".pt"), strict = False)
    except:
        return
def save_accuracies(new_acc, filename, alg):
    # if alg != "FISH" and alg != "BAIT":
    #     raise NameError("Invalid algorithm choice.")
    try:
        savefile = open("./Save/Round_accuracies/Accuracy_for_"+filename+'.p', "br")
        acc_dict = pickle.load(savefile)
        savefile.close()
    except:
        acc_dict = {"entropy": [], "kcent": [], "FISH":[],"BAIT":[], "rand" : [], "DUQ":[], "margin":[], "lcs":[], "FishEnt":[]}
    finally:
        if not os.path.exists("./Save/Round_accuracies"):
            os.makedirs("./Save/Round_accuracies")
        savefile = open("./Save/Round_accuracies/Accuracy_for_"+filename+'.p', "bw")
        acc_dict[alg].append(new_acc)
        pickle.dump(acc_dict, savefile)
        savefile.close()

def save_imp_weights(new_imp_wt_idxs, filename):
    try:
        savefile = open("./Save/Imp_weights/imp_wts_idxs_"+ filename+'.p', "br")
        imp_wt_idxs = pickle.load(savefile)
        savefile.close()
    except:
        imp_wt_idxs = []
    finally:
        if not os.path.exists("./Save/Imp_weights"):
            os.makedirs("./Save/Imp_weights")
        savefile = open("./Save/Imp_weights/imp_wts_idxs_"+ filename+'.p', "bw")
        imp_wt_idxs.append(new_imp_wt_idxs)
        pickle.dump(imp_wt_idxs, savefile)
        savefile.close()

def save_queried_idx(idx, filename, alg):
    try:
        savefile = open(f"./Save/Queried_idxs/{alg}_queried_idxs_"+ filename+'.p', "br")
        que_idxs = pickle.load(savefile)
        savefile.close()
    except:
        que_idxs = []
    finally:
        if not os.path.exists("./Save/Queried_idxs"):
            os.makedirs("./Save/Queried_idxs")
        savefile = open(f"./Save/Queried_idxs/{alg}_queried_idxs_"+ filename+'.p', "bw")
        que_idxs.append(idx)
        pickle.dump(que_idxs, savefile)
        savefile.close()


def save_dist_stats(stats_list, filename, alg):
    try:
        savefile = open(f"./Save/Queried_idxs/{alg}_diststats_"+ filename+'.p', "br")
        tE_stats = pickle.load(savefile)
        savefile.close()
    except:
        tE_stats = []
    finally:
        if not os.path.exists("./Save/Queried_idxs"):
            os.makedirs("./Save/Queried_idxs")
        savefile = open(f"./Save/Queried_idxs/{alg}_diststats_"+ filename+'.p', "bw")
        tE_stats.append(stats_list)
        pickle.dump(tE_stats, savefile)
        savefile.close()
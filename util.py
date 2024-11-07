import importlib

import torch
import numpy as np
from collections import abc

import multiprocessing as mp
from threading import Thread
from queue import Queue

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config, stored_refs = {}):
    if isinstance(config, dict):
        if "target" in config:
            params, ref_params, done = instantiate_from_config(config.get("params", dict()), stored_refs=stored_refs)

            if not done:
                return config, ref_params, False
            
            obj = get_obj_from_str(config["target"])(**params)

            d = ref_params

            if "ref" in config:
                ref_key = config["ref"]

                d[ref_key] = obj

            return obj, d, True
        elif "from_ref" in config:
            ref_key = config["from_ref"]
           
            if ref_key in stored_refs:
                d = {}
                if "ref" in config:
                    d[config["ref"]] = stored_refs[ref_key]
                return stored_refs[ref_key], d, True
            else:
                return config, {}, False
        elif "function_call" in config:

            obj, refs, done = instantiate_from_config(config["object"], stored_refs=stored_refs) 
            function_name = config["function_call"]

            if not done:
                return config, refs, False
            else:
                params, refs2, done2 = instantiate_from_config(config.get("params", dict()), stored_refs=stored_refs) 
                
                refs.update(refs2)

                if not done2:
                    return config, refs, False
                
                # call the function using params dict
                result = getattr(obj, function_name)(**params)

                return result, refs, True
        else:
            refs = {}
            d = {}
            total_done = True
            for key in config:
                d[key], sub_refs, done = instantiate_from_config(config[key], stored_refs=stored_refs)
                total_done = total_done and done
                refs.update(sub_refs)

            return d, refs, total_done
    elif isinstance(config, list):

        refs = {}
        d = []
        total_done = True
        for i in range(len(config)):
            d_i, sub_refs, done = instantiate_from_config(config[i], stored_refs=stored_refs)
            total_done = total_done and done
            refs.update(sub_refs)
            d.append(d_i)

        return d, refs, total_done

    else:
        return config, {}, True

def instantiate_object(config):
    obj, ref_params, done = instantiate_from_config(config)
    # print(ref_params.keys())
    while not done:
        obj, ref_params2, done = instantiate_from_config(obj, ref_params)
        ref_params.update(ref_params2)
        # print(ref_params.keys())
    return obj

def get_obj_from_str(string, reload=False):
    if string == "dict": return dict
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _do_parallel_data_prefetch(func, Q, data, idx, idx_to_fn=False):
    # create dummy dataset instance

    # run prefetching
    if idx_to_fn:
        res = func(data, worker_id=idx)
    else:
        res = func(data)
    Q.put([idx, res])
    Q.put("Done")


def parallel_data_prefetch(
        func: callable, data, n_proc, target_data_type="ndarray", cpu_intensive=True, use_worker_id=False
):
    # if target_data_type not in ["ndarray", "list"]:
    #     raise ValueError(
    #         "Data, which is passed to parallel_data_prefetch has to be either of type list or ndarray."
    #     )
    if isinstance(data, np.ndarray) and target_data_type == "list":
        raise ValueError("list expected but function got ndarray.")
    elif isinstance(data, abc.Iterable):
        if isinstance(data, dict):
            print(
                f'WARNING:"data" argument passed to parallel_data_prefetch is a dict: Using only its values and disregarding keys.'
            )
            data = list(data.values())
        if target_data_type == "ndarray":
            data = np.asarray(data)
        else:
            data = list(data)
    else:
        raise TypeError(
            f"The data, that shall be processed parallel has to be either an np.ndarray or an Iterable, but is actually {type(data)}."
        )

    if cpu_intensive:
        Q = mp.Queue(1000)
        proc = mp.Process
    else:
        Q = Queue(1000)
        proc = Thread
    # spawn processes
    if target_data_type == "ndarray":
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(np.array_split(data, n_proc))
        ]
    else:
        step = (
            int(len(data) / n_proc + 1)
            if len(data) % n_proc != 0
            else int(len(data) / n_proc)
        )
        arguments = [
            [func, Q, part, i, use_worker_id]
            for i, part in enumerate(
                [data[i: i + step] for i in range(0, len(data), step)]
            )
        ]
    processes = []
    for i in range(n_proc):
        p = proc(target=_do_parallel_data_prefetch, args=arguments[i])
        processes += [p]

    # start processes
    print(f"Start prefetching...")
    import time

    start = time.time()
    gather_res = [[] for _ in range(n_proc)]
    try:
        for p in processes:
            p.start()

        k = 0
        while k < n_proc:
            # get result
            res = Q.get()
            if res == "Done":
                k += 1
            else:
                gather_res[res[0]] = res[1]

    except Exception as e:
        print("Exception: ", e)
        for p in processes:
            p.terminate()

        raise e
    finally:
        for p in processes:
            p.join()
        print(f"Prefetching complete. [{time.time() - start} sec.]")

    if target_data_type == 'ndarray':
        if not isinstance(gather_res[0], np.ndarray):
            return np.concatenate([np.asarray(r) for r in gather_res], axis=0)

        # order outputs
        return np.concatenate(gather_res, axis=0)
    elif target_data_type == 'list':
        out = []
        for r in gather_res:
            out.extend(r)
        return out
    else:
        return gather_res



if __name__ == "__main__":
    
    # # test the dataset
    # folder = "data"

    # # create the dataset
    # dataset = WheelDataset(folder=folder)

    import cv2

    # now create the same dataset using instantiate_from_config
    config = {
        "target": "dataset.WheelDataset",
        "params": {
            "folder": "data"
        }
    }

    dataset = instantiate_object(config)

    # get the first positive example
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample["type"] == 1:

            # get bbox
            bbox = sample["bbox"]
            x, y, h, w = bbox

            # get image
            image = sample["image"]

            # draw the bbox
            image = cv2.rectangle(image, (y, x), (y+w, x+h), (255, 0, 0), 2)

            # show the image
            cv2.imshow("image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()









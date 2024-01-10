import os
import platform
from pathlib import Path
from subprocess import check_output

import pkg_resources as pkg
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
AUTOINSTALL = str(os.getenv("YOLOv5_AUTOINSTALL", True)).lower() == "true"  # global auto-install mode
FONT = "Arial.ttf"  # https://ultralytics.com/assets/Arial.ttf


def try_except(func):
    # try-except function. Usage: @try_except decorator
    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e)

    return handler


def is_ascii(s=""):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode("ascii", "ignore")) == len(s)


def is_writeable(dir, test=False):
    # Return True if directory has write permissions, test opening a file with write permissions if test=True
    if not test:
        return os.access(dir, os.W_OK)  # possible issues on Windows
    file = Path(dir) / "tmp.txt"
    try:
        with open(file, "w"):  # open file with write permissions
            pass
        file.unlink()  # remove file
        return True
    except OSError:
        return False


def user_config_dir(dir="Ultralytics", env_var="YOLOV5_CONFIG_DIR"):
    # Return path of user configuration directory. Prefer environment variable if exists. Make dir if required.
    env = os.getenv(env_var)
    if env:
        path = Path(env)  # use environment variable
    else:
        cfg = {
            "Windows": "AppData/Roaming",
            "Linux": ".config",
            "Darwin": "Library/Application Support",
        }  # 3 OS dirs
        path = Path.home() / cfg.get(platform.system(), "")  # OS-specific config dir
        path = (
            path if is_writeable(path) else Path("/tmp")
        ) / dir  # GCP and AWS lambda fix, only /tmp is writeable
    path.mkdir(exist_ok=True)  # make if required
    return path


CONFIG_DIR = user_config_dir()


def check_font(font=FONT, progress=False):
    # Download font to CONFIG_DIR if necessary
    font = Path(font)
    file = CONFIG_DIR / font.name
    if not font.exists() and not file.exists():
        url = "https://ultralytics.com/assets/" + font.name
        print(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file), progress=progress)


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def check_python(minimum="3.7.0"):
    # Check current python version vs. required python version
    check_version(platform.python_version(), minimum, name="Python ", hard=True)


def check_version(
    current="0.0.0",
    minimum="0.0.0",
    name="version ",
    pinned=False,
    hard=False,
    verbose=False,
):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f"{name}{minimum} required by YOLOv5, but {name}{current} is currently installed"  # string
    if hard:
        assert result, s  # assert min requirements met
    if verbose and not result:
        print(s)
    return result


def check_online():
    # Check internet connectivity
    import socket

    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False


@try_except
def check_requirements(requirements=ROOT / "requirements.txt", exclude=(), install=True, cmds=()):
    # Check installed dependencies meet requirements (pass *.txt file or list of packages)
    prefix = colorstr("red", "bold", "requirements:")
    check_python()  # check python version
    if isinstance(requirements, (str, Path)):  # requirements.txt file
        file = Path(requirements)
        assert file.exists(), f"{prefix} {file.resolve()} not found, check failed."
        with file.open() as f:
            requirements = [
                f"{x.name}{x.specifier}" for x in pkg.parse_requirements(f) if x.name not in exclude
            ]
    else:  # list or tuple of packages
        requirements = [x for x in requirements if x not in exclude]

    n = 0  # number of packages updates
    for i, r in enumerate(requirements):
        try:
            pkg.require(r)
        except Exception:  # DistributionNotFound or VersionConflict if requirements not met
            s = f"{prefix} {r} not found and is required by YOLOv5"
            if install and AUTOINSTALL:  # check environment variable
                print(f"{s}, attempting auto-update...")
                try:
                    assert check_online(), f"'pip install {r}' skipped (offline)"
                    print(check_output(f'pip install "{r}" {cmds[i] if cmds else ""}', shell=True).decode())
                    n += 1
                except Exception as e:
                    print(f"{prefix} {e}")
            else:
                print(f"{s}. Please install and rerun your command.")

    if n:  # if packages updated
        source = file.resolve() if "file" in locals() else requirements
        s = (
            f"{prefix} {n} package{'s' * (n > 1)} updated per {source}\n"
            f"{prefix} ⚠️ {colorstr('bold', 'Restart runtime or rerun command for updates to take effect')}\n"
        )
        print(s)


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

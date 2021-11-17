import os
import re

def validate_file_path(path:str, ext:str=""):
    """
    Validate file path -  whether
    base path exists, 
    file name has correct extension [verified only in case ext passed],
    enforces naming conventions on basename only contain characters [a-zA-Z0-9_], length of 255
    (https://www.ibm.com/docs/en/aix/7.1?topic=files-file-naming-conventions)
    
    args
    - path             path whose validity is tb verified
    - ext              file extension - validity of the extension not verified 

    returns
    - str              error message - empty if path valid         
    """
    base_path, head = os.path.split(os.path.abspath(path))
    
    if not os.path.isdir(base_path):
        return f"Basepath { base_path } of parameter path is not a valid directory."
    if not head.endswith(ext):
        return f"Filename { head } does not have the correct extension { ext }."
    fname = head
    if ext:
        fname = head[:-len(ext)]
    
    if len(fname) > (255 - len(ext)) or not re.fullmatch("[a-zA-Z0-9_]+", fname):
        return f"Base file name { fname } must be of length <= 255 and contain only characters [a-zA-Z0-9_]."
    return ""

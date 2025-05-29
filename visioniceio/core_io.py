"""Core IO functions for VisionICe data files.

These functions provide the basic reading capabilities for the binary files. 
They are used by the Experiment class, in which also the loading structure is 
defined.
"""

import struct

import numpy as np


dtype_map = {
    'uint32': ('>u4', 4),
    'int32' : ('>i4', 4),
    'uint16': ('>u2', 2),
    'int16' : ('>i2', 2)
}


def read_metadata(filepath):
    """
    Parses metadata from a text file.

    This function reads a text file and extracts metadata in the form of key-value pairs. 
    It processes boolean values, numeric values, and comma-separated lists of numbers.

    Args:
        filepath (str): The path to the metadata file.

    Returns:
        dict: A dictionary containing the parsed metadata.

    Note:
        Currently, this function is designed to parse metadata from a `.txt` file. 
        It should later be adapted to handle `.ifo` files.
    """
    result = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # skip empty lines
            line = line.strip()
            if not line or ':' not in line:
                continue

            key, val = map(str.strip, line.split(':', 1))

            # Handle yes/no as booleans
            lower = val.lower()
            if lower in ('yes', 'true'):
                parsed = True
            elif lower in ('no', 'false'):
                parsed = False

            # Handle comma-separated lists of numbers
            elif ',' in val:
                items = [item.strip() for item in val.split(',')]
                # try converting all to int
                try:
                    parsed = [int(item) for item in items]
                except ValueError:
                    # try floats if ints fail
                    try:
                        parsed = [float(item.replace(',', '.')) for item in items]
                    except ValueError:
                        parsed = items

            # Handle single numeric values
            else:
                # try int
                try:
                    parsed = int(val)
                except ValueError:
                    # try float (with comma as decimal separator)
                    try:
                        parsed = float(val.replace(',', '.'))
                    except ValueError:
                        # fallback to string
                        parsed = val

            result[key] = parsed

    return result


def read_data(filename, dtype, nd):
    """Reading the Binary Lab Files.
    
    Args:
        filename (str): Path to the file to be read.
        dtype (str): Data type of the file. Supported types are 'int16', 'int32', 'float32', 'float64'.
        nd (int): Number of dimensions of one single trial data set.

    Returns:
        data (list): List of numpy arrays, each containing one dataset.

    Note:
        A dataset is here defined as the appropiate data for one trial.
        For an electrode trace it is a 1d object (the one electrode.)
        For waveforms it is a 2d object (index of the spike, signal of spike)
    """
    # functions input assertion
    if dtype not in dtype_map:
            raise ValueError(f"Unsupported datatype {dtype}")
    np_dtype, datasize = dtype_map[dtype]
    
    with open(filename, 'rb') as f:

        # file header and metadata
        header = f.read(4).decode('ascii')
        if header != 'DTLG':
            raise ValueError(f"Unsupported file, expected header 'DTLG', got '{header}'")
        
        version = f.read(4)#.decode('ascii')
        # print(version)
        ndim = struct.unpack('>I', f.read(4))[0]
        # print(f"Number of datasets: {ndim}")
        p = struct.unpack('>I', f.read(4))[0]
        # print(f"Offset to datasets: {p}")
        ld = struct.unpack('>h', f.read(2))[0]
        # print(f"Length of descriptor: {ld}")
        descriptor = f.read(ld).decode('ascii')
        # print(f"Descriptor: {descriptor}")

        # load offsets to each dataset
        f.seek(p)
        if ndim <= 128:
            raw = f.read(128 * 4)
            offs = np.frombuffer(raw, dtype='>u4')[:ndim]
        else:
            # recursive loading of multiple blocks
            offs = []
            block = np.frombuffer(f.read(128 * 4), dtype='>u4')
            offs.extend(block[block > 0].tolist())
            while len(offs) < ndim:
                new_offs = []
                for off in offs:
                    f.seek(int(off))
                    raw = f.read(128 * 4)
                    blk = np.frombuffer(raw, dtype='>u4')
                    new_offs.extend(blk[blk > 0].tolist())
                offs = new_offs
            offs = offs[:ndim]
        offset = np.array(offs, dtype=np.uint32)

        # read each dataset
        data = []
        for off in offset:
            f.seek(int(off))
            # read dimension sizes (C-order) and flip to match MATLAB fliplr
            dims = struct.unpack('>' + 'i'*nd, f.read(4 * nd))
            count = int(np.prod(dims))
            raw = f.read(count * datasize)
            arr = np.frombuffer(raw, dtype=np_dtype, count=count).astype(dtype=dtype)
            arr = arr.reshape(dims)
            data.append(arr)

        return data
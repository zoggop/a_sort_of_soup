import os
import base64
import traceback
import pickle
from pbkdf2 import PBKDF2
from Crypto.Cipher import AES
import string
import random

# adapted from:
# https://infotechbrain.com/2018/09/examples-of-python-password-encryption-stored-in-a-file/

class Catacomb:
    PASSPHRASE_SIZE = 64  # 512-bit passphrase
    KEY_SIZE = 32  # 256-bit key
    BLOCK_SIZE = 16  # 16-bit blocks
    IV_SIZE = 16  # 128-bits to initialize
    SALT_SIZE = 8  # 64-bits of salt
    SEED_SIZE = 18 # 18 byte random seed string

    def __init__(self, directory, seed=None):
        self.KP_FILE = '{}/kfileNsxConfigVerfiy.p'.format(directory)
        self.SDB_FILE = '{}/sdbfileNsxConfigVerify'.format(directory)
        self.SEED_FILE = '{}/catacomb.seed'.format(directory)
        if seed == 'reset':
            # erase all stored data and create a new seed
            os.remove(self.KP_FILE)
            os.remove(self.SDB_FILE)
            os.remove(self.SEED_FILE)
            seed = None
        if seed is None:
            if os.path.exists(self.SEED_FILE):
                with open(self.SEED_FILE, 'r') as seedfile:
                    seed = seedfile.read().strip()
            if seed is None or seed == '':
                seed = ''.join(random.choice(string.ascii_letters + string.digits) for i in range(self.SEED_SIZE))
                with open(self.SEED_FILE, 'w') as seedfile:
                    seedfile.write(seed)
        self.directory = directory
        self.SEED = seed
        try:
            with open(self.KP_FILE, 'rb') as f:
                self.kp = f.read()
            if len(self.kp) == 0: raise IOError
        except IOError:
            with open(self.KP_FILE, 'wb') as f:
                # Generate Random kp
                self.kp = os.urandom(self.PASSPHRASE_SIZE)
                f.write(base64.b64encode(self.kp))

                try:
                    # If the kp has to be regenerated, then the old data in the SDB file can no longer be used and should be removed
                    if os.path.exists(self.SDB_FILE):
                        os.remove(self.SDB_FILE)
                except:
                    print(traceback.format_exc())
                    print("There might be an error with permissions for the SDB_FILE {}".format(self.SDB_FILE))
        else:
            # decode from base64
            self.kp = base64.b64decode(self.kp)

        # Load or create SDB_FILE:
        try:
            with open(self.SDB_FILE, 'rb') as f:
                self.sdb = pickle.load(f)
            # sdb will be a dictionary that will have key, value pairs
            if self.sdb == {}: raise IOError
        except (IOError, EOFError):
            self.sdb = {}
            with open(self.SDB_FILE, 'wb') as f:
                pickle.dump(self.sdb, f)

    def getSaltForPname(self, pname):
        # Salt is generated as the hash of the key with it's own salt acting like a seed value
        return PBKDF2(pname, self.SEED).read(self.SALT_SIZE)

    # Encrypt Password
    def encrypt(self, pname, p):
        # Pad p, then encrypt it with a new, randomly initialised cipher.  Will not preserve trailing whitespace in plaintext!

        # Initialise Cipher Randomly
        initVector = os.urandom(self.IV_SIZE)

        salt = self.getSaltForPname(pname)

        # Prepare cipher key that will be used to encrypt and decrypt
        k = PBKDF2(self.kp, salt).read(self.KEY_SIZE)

        # Create cipher that will be used to encrypt the data
        cipher = AES.new(k, AES.MODE_CBC, initVector)

        # Pad and encrypt
        self.sdb[pname] = initVector + cipher.encrypt(bytes(p + ' '*(self.BLOCK_SIZE - (len(p) % self.BLOCK_SIZE)), 'utf-8'))
        with open(self.SDB_FILE, 'wb') as f:
            pickle.dump(self.sdb, f)

    # Decrypt Password
    def decrypt(self, pname):
        # Reconstruct the cipher object and decrypt. Will not preserve trailing whitespace in the retrieved value!
        if self.sdb.get(pname) == None:
            return None

        salt = self.getSaltForPname(pname)

        # Recreate an identical cipher key:
        key = PBKDF2(self.kp, salt).read(self.KEY_SIZE)

        # Get initVector (salt) that was concatenated into the encrypted Data stored in the SDB_FILE
        initVector = self.sdb[pname][:self.IV_SIZE]

        # Get only the data you want to decrypt
        encryptedData = self.sdb[pname][self.IV_SIZE:]

        # Recreate cipher
        cipher = AES.new(key, AES.MODE_CBC, initVector)

        # Decrypt, depad, and decode
        return cipher.decrypt(encryptedData).rstrip(bytes(' ', 'utf-8')).decode('utf-8')
import numpy as np
import os

class Element:
    """Element class containing atomic properties"""
    
    # Class variable to store periodic table data
    _periodic_table = {}
    
    def __init__(self, symbol=None, number=None):
        """
        Initialize an element by symbol or atomic number
        
        Args:
            symbol: Element symbol (e.g., 'H', 'He')
            number: Atomic number (e.g., 1, 2)
        """
        # Load periodic table data if not already loaded
        self._load_periodic_table()
        
        if symbol is not None:
            self._init_by_symbol(symbol)
        elif number is not None:
            self._init_by_number(number)
        else:
            raise ValueError("Either symbol or number must be provided")

    @classmethod
    def _load_periodic_table(cls):
        """Load periodic table data from CSV file"""
        if cls._periodic_table:
            return
        
        data_file = os.path.join(os.path.dirname(__file__), 'data', 'periodic_table.csv')
        
        # Check if file exists
        if not os.path.exists(data_file):
            print(f"Warning: {data_file} not found. Using fallback data.")
            cls._init_fallback_data()
            return
        
        # Load the CSV file
        data_all = np.loadtxt(data_file, delimiter=',', dtype=str)
        data_headers = data_all[0]  # Get headers
        data_type = data_all[1]  # Get data rows
        data = data_all[2:]
        
        header_to_index = {header: i for i, header in enumerate(data_headers)}
        
        for row in data:
            element_data = {}
            
            # Automatically process each column based on header and data type
            for header, dtype in zip(data_headers, data_type):
                value = row[header_to_index[header]]
                
               # Skip empty values
                if value == '' or value is None:
                    element_data[str(header)] = None
                    continue
                
                # Convert based on data type
                try:
                    value = eval(dtype)(value)   
                    element_data[str(header)] = value
                except ValueError:
                    # If conversion fails, keep as string
                    element_data[str(header)] = value                

            # Store by both symbol and atomic number
            cls._periodic_table[element_data['symbol']] = element_data
            cls._periodic_table[element_data['number']] = element_data        
            
    def _init_by_symbol(self, symbol):
        """Initialize by element symbol"""
        if symbol not in self._periodic_table:
            raise ValueError(f"Element '{symbol}' not found in periodic table")
        
        data = self._periodic_table[symbol]
        for key, value in data.items():
            setattr(self, key, value)
         
    def _init_by_number(self, number):
        """Initialize by atomic number"""
        if number not in self._periodic_table:
            raise ValueError(f"Atomic number {number} not found in periodic table")
        
        data = self._periodic_table[number]
        for key, value in data.items():
            setattr(self, key, value)
    
class Atom(object):
    
    def __init__(self, element, coordinate):
        """
        Initialize an atom
        
        Parameters
        ----------
        element : Element or str or int
            Can be an Element object, element symbol (str), or atomic number (int)
        coordinate : list
            [x, y, z] coordinates of the atom
        """
        # Handle different input types for element
        if isinstance(element, Element):
            # Already an Element object
            for name, value in element.__dict__.items():
                setattr(self, name, value)
        elif isinstance(element, str):
            # Element symbol
            elem = Element(symbol=element)
            for name, value in elem.__dict__.items():
                setattr(self, name, value)
        elif isinstance(element, int):
            # Atomic number
            elem = Element(number=element)
            for name, value in elem.__dict__.items():
                setattr(self, name, value)
        else:
            raise TypeError(f"element must be Element, str, or int, not {type(element)}")
        
        self.assign_coordinate(coordinate)

    def assign_coordinate(self, coordinate):
        """
        Assign coordinates to the atom
        
        Parameters
        ----------
        coordinate : list
            [x, y, z] coordinates
        
        Raises
        ------
        TypeError
            If coordinate is not a list
        ValueError
            If coordinate length is not 3
        """
        if not isinstance(coordinate, list):
            raise TypeError("coordinate must be list")
        if not len(coordinate) == 3:
            raise ValueError("length of the coordinate must be 3")
        self.coordinate = np.array(coordinate, dtype=float)

    def __eq__(self, other, error=1e-5):
        """Check if two atoms are equal"""
        if not isinstance(other, Atom):
            return False
        if self.number != other.number:
            return False
        if not np.allclose(self.coordinate, other.coordinate, error):
            return False
        return True
        
class Molecule(object):
    """
    Molecule class representing a collection of atoms
    """   
    def __init__(self, atoms=None, charge=0, multiplicity=1):
        """
        Initialize a molecule
        
        Parameters
        ----------
        atoms : list, optional
            List of Atom objects
        charge : int, optional
            Total molecular charge (default: 0)
        multiplicity : int, optional
            Spin multiplicity (default: 1 for singlet)
        """
        self.atoms = []
        self.charge = charge
        self.multiplicity = multiplicity
        
        if atoms:
            for atom in atoms:
                self.add_atom(atom)
    
    def add_atom(self, atom):
        """Add an atom to the molecule"""
        if not isinstance(atom, Atom):
            raise TypeError("Can only add Atom objects")
        self.atoms.append(atom)
    
    def remove_atom(self, index):
        """Remove atom at given index"""
        if index < 0 or index >= len(self.atoms):
            raise IndexError(f"Atom index {index} out of range")
        return self.atoms.pop(index)
    
    def get_atom(self, index):
        """Get atom at given index"""
        if index < 0 or index >= len(self.atoms):
            raise IndexError(f"Atom index {index} out of range")
        return self.atoms[index]
    
    @property
    def coordinate(self):
        """Get all atomic coordinates as Nx3 array"""
        return np.array([atom.coordinate for atom in self.atoms])
    
    @property
    def mass(self):
        """Get total molecular mass"""
        return sum(atom.mass for atom in self.atoms)
    
    def __len__(self):
        return len(self.atoms)
    
    def __getitem__(self, index):
        return self.get_atom(index)
    
    def __iter__(self):
        return iter(self.atoms)
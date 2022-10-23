from operator import itemgetter
import numpy as np


class Subdomain:
    def __init__(self, id, mat, borders=None) -> None:
        self.id = id
        self.mat = mat
        self.borders = borders


class Subdomains:
    def __init__(self, subdomains) -> None:
        self.subdomains = subdomains

    def check_unique_names(self):
        # check that ids are different
        mat_names = []
        for subdomain in self.subdomains:
            if type(subdomain.mat.name) is list:
                mat_names += subdomain.mat.name
            else:
                mat_names.append(subdomain.mat.name)

        if len(mat_names) != len(np.unique(mat_names)):
            raise ValueError("Some materials have the same name")

    def check_borders(self, size):
        """Checks that the borders of the materials match
        Args:
            size (float): size of the 1D domain
        Raises:
            ValueError: if the borders don't begin at zero
            ValueError: if borders don't match
            ValueError: if borders don't end at size
        Returns:
            bool -- True if everything's alright
        """
        all_borders = []
        for m in self.subdomains:
            if isinstance(m.borders[0], list):
                for border in m.borders:
                    all_borders.append(border)
            else:
                all_borders.append(m.borders)
        all_borders = sorted(all_borders, key=itemgetter(0))
        if all_borders[0][0] != 0:
            raise ValueError("Borders don't begin at zero")
        for i in range(0, len(all_borders)-1):
            if all_borders[i][1] != all_borders[i+1][0]:
                raise ValueError("Borders don't match to each other")
        if all_borders[len(all_borders) - 1][1] != size:
            raise ValueError("Borders don't match with size")
        return True

    def find_subdomain_from_x_coordinate(self, x):
        """Finds the correct subdomain at a given x coordinate
        Args:
            x (float): the x coordinate
        Returns:
            int: the corresponding subdomain id
        """
        for subdomain in self.subdomains:
            # if no borders are provided, assume only one subdomain
            if subdomain.borders is None:
                return subdomain.id
            # else find the correct material
            else:
                if isinstance(subdomain.borders[0], list) and \
                        len(subdomain.borders) > 1:
                    list_of_borders = subdomain.borders
                else:
                    list_of_borders = [subdomain.borders]
                if isinstance(subdomain.id, list):
                    ids = subdomain.id
                else:
                    ids = [
                        subdomain.id for _ in range(len(list_of_borders))]

                for borders, id_ in zip(list_of_borders, ids):
                    if borders[0] <= x <= borders[1]:
                        return id_
        # if no subdomain was found, return 0
        return 0
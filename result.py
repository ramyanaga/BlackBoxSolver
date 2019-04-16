class Result:
    def __init__(self, args):
    """
    Args:
        self.distortions: list
                FINISH
        self.query_counts: list
                FINISH
        self.flag: list
                FINISH
    """


        self.distortions = args['distortions']
        self.query_counts = args['query_count']
        self.flag = [] # set to 1 if iteration created succesful attack, 0 otherwise
        self.perturbed_image = args['image_id'] # need to flatten first, only hold best attack

import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        for l in range(y_probs.shape[1]):
            max_ind = np.argmax(y_probs[:,l,0])
            path_prob *= y_probs[max_ind,l,0]
            decoded_path.append(max_ind)
        
        decoded_path = ''.join([self.symbol_set[decoded_path[i]-1] for i in range(len(decoded_path)) if decoded_path[i] != blank and ((i == 0) or (i > 0 and decoded_path[i] != decoded_path[i-1]))])

        return decoded_path, path_prob


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        # SymbolSet: a list of symbols not including the blank
        self.symbol_set = symbol_set
        self.beam_width = beam_width
        # PathScore: dictionary of scores for paths ending with symbols
        self.PathScore = {}
        # BlankPathScore: dictionary of scores for paths ending with blanks
        self.BlankPathScore = {}
        self.blank = 0
        self.scorelist = []

    def InitializePaths(self, SymbolSet, y):
        InitialBlankPathScore = {}
        InitialPathScore = {}
        # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
        path = ''
        InitialBlankPathScore[path] = y[self.blank] # Score of blank at t=1 
        # print(y[])
        InitialPathsWithFinalBlank = {path}
        # Push rest of the symbols into a path-ending-with-symbol stack
        InitialPathsWithFinalSymbol = set()
        for c in SymbolSet: # This is the entire symbol set, without the blank
            path = c
            p = SymbolSet.index(path)+1
            InitialPathScore[path] = y[p] # Score of symbol c at t=1
            InitialPathsWithFinalSymbol.add(path) # Set addition
        
        return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

    def Prune(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
        PrunedBlankPathScore = {}
        PrunedPathScore = {}
        # First gather all the relevant scores
        scorelist = []
        for p in PathsWithTerminalBlank:
            scorelist.append(BlankPathScore[p])
        for p in PathsWithTerminalSymbol:
            scorelist.append(PathScore[p])
        
        # Sort and find cutoff score that retains exactly BeamWidth paths
        scorelist = sorted(scorelist,reverse=True)
        # print(scorelist)
        cutoff = scorelist[BeamWidth-1] if BeamWidth < len(scorelist) else scorelist[-1]
        # print(BeamWidth)
        # print(cutoff)
        
        PrunedPathsWithTerminalBlank = set()
        for p in PathsWithTerminalBlank:
            if BlankPathScore[p] >= cutoff:
                PrunedPathsWithTerminalBlank.add(p) # Set addition
                PrunedBlankPathScore[p] = BlankPathScore[p]
        
        PrunedPathsWithTerminalSymbol = set()
        for p in PathsWithTerminalSymbol:
            if PathScore[p] >= cutoff:
                PrunedPathsWithTerminalSymbol.add(p) # Set addition 
                PrunedPathScore[p] = PathScore[p]
        
        return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

    def ExtendWithBlank(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
        UpdatedPathsWithTerminalBlank = set()
        UpdatedBlankPathScore = {}
        # First work on paths with terminal blanks
        #(This represents transitions along horizontal trellis edges for blanks)
        for path in PathsWithTerminalBlank:
            # Repeating a blank doesn’t change the symbol sequence
            UpdatedPathsWithTerminalBlank.add(path) # Set addition
            UpdatedBlankPathScore[path] = self.BlankPathScore[path]*y[self.blank]
        
        # Then extend paths with terminal symbols by blanks
        for path in PathsWithTerminalSymbol:
            # If there is already an equivalent string in UpdatesPathsWithTerminalBlank
            # simply add the score. If not create a new entry
            if path in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[path] += self.PathScore[path]* y[self.blank]
            else:
                UpdatedPathsWithTerminalBlank.add(path) # Set addition
                UpdatedBlankPathScore[path] = self.PathScore[path] * y[self.blank]
        
        return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

    def ExtendWithSymbol(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y):
        UpdatedPathsWithTerminalSymbol = set()
        UpdatedPathScore = {}
        
        # First extend the paths terminating in blanks. This will always create a new sequence
        for path in PathsWithTerminalBlank:
            for c in SymbolSet: # SymbolSet does not include blanks
                newpath = path + c # Concatenation
                UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition
                UpdatedPathScore[newpath] = self.BlankPathScore[path] * y[SymbolSet.index(c)+1]
        
        # Next work on paths with terminal symbols
        for path in PathsWithTerminalSymbol:
            # Extend the path with every symbol other than blank
            for c in SymbolSet: # SymbolSet does not include blanks
                newpath = path if (c == path[-1]) else path + c # Horizontal transitions don’t extend the sequence
                if newpath in UpdatedPathsWithTerminalSymbol: # Already in list, merge paths
                    UpdatedPathScore[newpath] += self.PathScore[path] * y[SymbolSet.index(c)+1]
                else: # Create new path
                    UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition
                    UpdatedPathScore[newpath] = self.PathScore[path] * y[SymbolSet.index(c)+1]
                    
        return UpdatedPathsWithTerminalSymbol, UpdatedPathScore
    
    def MergeIdenticalPaths(self, PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
        # All paths with terminal symbols will remain
        MergedPaths = PathsWithTerminalSymbol 
        FinalPathScore = PathScore
        # Paths with terminal blanks will contribute scores to existing identical paths from # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
        for p in PathsWithTerminalBlank:
            if p in MergedPaths:
                FinalPathScore[p] += BlankPathScore[p]
            else:
                MergedPaths.add(p) # Set addition
                FinalPathScore[p] = BlankPathScore[p]

        return MergedPaths, FinalPathScore

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """
        
        # First time instant: Initialize paths with each of the symbols, 
        # including blank, using score at time t=1
        NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = self.InitializePaths(self.symbol_set, y_probs[:,0,0])
        print("INIT:")
        print('path w/ terminal blank: '+str(NewPathsWithTerminalBlank))
        print('path w/ terminal symbol: '+str(NewPathsWithTerminalSymbol))
        print('blank path score: '+str(NewBlankPathScore))
        print('path score: '+str(NewPathScore))
        T = y_probs.shape[1]-1
        
        for t in range(T):
            # Prune the collection down to the BeamWidth
            y = y_probs[:,t+1,0]
            PathsWithTerminalBlank, PathsWithTerminalSymbol, self.BlankPathScore, self.PathScore = self.Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore, self.beam_width)
            # print('t = '+str(t+1))
            # print("Prune:")
            # print('path w/ terminal blank: '+str(PathsWithTerminalBlank))
            # print('path w/ terminal symbol: '+str(PathsWithTerminalSymbol))
            # print('blank path score: '+str(self.BlankPathScore))
            # print('path score: '+str(self.PathScore))
            
            # First extend paths by a blank
            NewPathsWithTerminalBlank, NewBlankPathScore = self.ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y)
            
            # Next extend paths by a symbol
            NewPathsWithTerminalSymbol, NewPathScore = self.ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, self.symbol_set, y)
            # print('\n')
            # print("Extend w/ blanks and symbols:")
            # print('path w/ terminal blank: '+str(NewPathsWithTerminalBlank))
            # print('blank path score: '+str(NewBlankPathScore))
            # print('path w/ terminal symbol: '+str(NewPathsWithTerminalSymbol))
            # print('path score: '+str(NewPathScore))
            
        # Merge identical paths differing only by the final blank
        MergedPaths, FinalPathScore = self.MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, NewPathsWithTerminalSymbol, NewPathScore)
        print("FINAL: "+str(FinalPathScore))
        #Pick best path
        bestPath = max(FinalPathScore,key=FinalPathScore.get)

        return bestPath, FinalPathScore

#  A group of useful functions that don't belong to anything in particular

import subprocess as SP
import numpy as N
import csv
def write_mat_text(A,filename,delimiter=','):
    """
    Writes a matrix to file, 1D or 2D, in text with delimeter and a space
    seperating the elements.
    """

    if len(N.shape(A))>2:
        raise RuntimeError('Can only write matrices with 1 or 2 dimensions')
        
    A = N.mat(A)
    numRows,numCols = N.shape(A) #must be 2D since it is a matrix

    writer = csv.writer(open(filename,'w'),delimiter=delimiter)
    
    #def genStringElement(
    
    for rowNum in range(numRows):
        row=[]
        for colNum in range(numCols):
            row.append(str(A[rowNum,colNum]))
        writer.writerow(row)
    
def read_mat_text(filename,delimiter=','):
    """ Reads a matrix written by write_mat_text, plain text"""
    import csv
    f = open(filename,'r')
    matReader = csv.reader(f,delimiter=delimiter)
    #read the entire file first to get dimensions.
    numLines = 0
    for line in matReader:
        numLines+=1
        if numLines ==1:
            lineLength = len(line)
    #rewind to beginning of file and read again
    f.seek(0)
    A = N.zeros((numLines,lineLength))
    for i,line in enumerate(matReader):
        A[i,:] =  N.array([float(j) for j in line])
    return A

def find_file_type(filename):
    l = len(filename)
    n=-1
    while abs(n)<l and filename[n]!='.':
        n-=1
    fileExtension = filename[n+1:]
    return fileExtension

def get_file_list(dir,fileExtension=None):
    """ Finds all files in the given directory that have the given file extension"""
    filesRaw = SP.Popen(['ls',dir],stdout=SP.PIPE).communicate()[0]
    #files separated by endlines
    filename= ''
    fileList=[]
    #print 'filesRaw is ',filesRaw
    for c in filesRaw:
        if c!='\n':
            filename+=c
        else: #completed file name
            if fileExtension is not None and filename[-len(fileExtension):] == fileExtension:
                fileList.append(filename)
            elif fileExtension is None:
                fileList.append(filename)
            filename=''
    return fileList
  
class MPI(object):
    """Simple container for information about how many processors there are.
    It ensures no failure in case mpi4py is not installed or running serial."""
    def __init__(self,numCPUs=None):
        try:
            from mpi4py import MPI as MPI_mod
            self.comm = MPI_mod.COMM_WORLD
            if (numCPUs is None) or (numCPUs > self.comm.Get_size()) or \
            (numCPUs<=0): 
                self.numCPUs = self.comm.Get_size()
            else: #use fewer CPUs than are available
                self.numCPUs = numCPUs      
            self.rank = self.comm.Get_rank()
        except ImportError:
            self.numCPUs=1
            self.rank=0
            self.comm = None
   
    def find_consec_proc_assignments(self,numTasks):
        """Finds the tasks for each processor, giving the tasks numbers
        from 0 to numTasks-1. 
        
        It assumes the tasks can be numbered consecutively,
        hence the name. The returned list is numCPUs+1 long and contains
        the assignments as:
        Proc n has tasks taskProcAssignments[n:(n+1)]
        """
        taskProcAssignments=[]
        #In the future it would be better to reevaulate
        #how many tasks per proc formula. When there are more than
        #half as many procs as tasks, almost half of the procs do 
        # nothing. Must change tests if this is done
        numTasksPerProc = int(N.ceil(numTasks/(1.*self.numCPUs)))
        for procNum in range(self.numCPUs+1):
            if procNum*numTasksPerProc <= numTasks:
                taskProcAssignments.append(procNum*numTasksPerProc)
            else:
                taskProcAssignments.append(numTasks)
        return taskProcAssignments
        
    def find_proc_assignments(self,taskList):
      """ Finds the breakdown of tasks for each processor, evenly
      breaking up the tasks in the taskList. It returns a list
      that has numCPUs+1 entries. 
      Proc n is responsible for taskProcAssignments[n][...]
      where the 2nd dimension of the 2D list contains the tasks (whatever
      they were in the original taskList)
      """
      
      numTasks = len(taskList)
      numTasksPerProc = int(N.ceil(numTasks/(1.*self.numCPUs)))
      taskProcAssignments= []
      for procNum in range(self.numCPUs):
          if (procNum+1)*numTasksPerProc < numTasks:
              taskProcAssignments.append(\
                taskList[procNum*numTasksPerProc:(procNum+1)*numTasksPerProc])
          else:
              taskProcAssignments.append(taskList[procNum*numTasksPerProc:])
      return taskProcAssignments
      



        
  

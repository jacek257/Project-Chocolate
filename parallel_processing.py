#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 16:27:11 2019

@author: Jimi Cao
"""
import subprocess
import os
import signal
import sys
import time

def kill_unending(processes, verb):
    '''
    Kills processes if flirt has been running for too long so the FEAT can continue
    
    Parameters:
        processes : array-like
            List of processes that are being run
        verb : boolean
            Flag to turn on verbose output
            
    Returns:
        Nothing
    '''
    proc = subprocess.Popen("ps -e | grep flirt", encoding='utf-8', stdout=subprocess.PIPE, shell=True)

    outs = proc.communicate()[0].split('\n')
    for line in outs:
        parts = line.split()
        if len(parts) > 0:
            time = parts[2].split(':')
            secs = int(time[0])*3600 + int(time[1])*60 + int(time[2])
            if secs > 900:
                if verb:
                    print('Killing process, flirt taking too long')
                os.kill(int(parts[0]), signal.SIGTERM)
    
    return

def get_next_avail(processes, verb, limit=5, key, s_name):
    '''
    Managues the queue for processes
    
    Parameters:
        processes : array-like
            List of processes that are being run
        verb : boolean
            Flag to turn on verbose output
        limit : int
            Number of processes that can be ran at once
        key : str
            What type of analysis is being done
        s_name : str
            Name of the script that is being run
    
    Returns:
        index : int
            Index of the in the processing queue that has become open
    '''
    
    msg = False
    spin = '|/-\\'
    cursor = 0
    while not any(v is None for v in processes):
        if verb:
            if not msg:
                print('There are', limit, key, s_name, 'currently running. Limit reached. Waiting for at least one to end.')
                msg = True
            else:
                sys.stdout.write(spin[cursor])
                sys.stdout.flush()
                cursor += 1
                if cursor >= len(spin):
                    cursor = 0
        
        kill_unending(processes, verb)
        
        for i, process in enumerate(processes):
            if process != None and process.poll() != None:
                processes[i] = None
                break
                
        if verb:
            if msg:
                time.sleep(0.2)
                sys.stdout.write('\b')
    
    return processes.index(None)

def wait_remaining(processes, verb, key, s_name):
    '''
    Wait for the queue to empty
    
    Parameters:
        processes : array-like
            List of processes that are being run
        verb : boolean
            Flag to turn on verbose output
        key : str
            What type of analysis is being done
        s_name : str
            Name of the script that is being run
    
    Returns:
        Nothing
    '''
    
    msg = False
    spin = '|/-\\'
    cursor = 0
        
    while not all(v is None for v in processes):
        if verb:
            if not msg:
                print('Waiting for the remaining', key, s_name, 'to finish')
                msg = True
            else:
                sys.stdout.write(spin[cursor])
                sys.stdout.flush()
                cursor += 1
                if cursor >= len(spin):
                    cursor = 0
        
        kill_unending(processes, verb)
        
        for i, process in enumerate(processes):
            if process != None and process.poll() != None:
                processes[i] = None
                    
        if verb:
            if msg:
                time.sleep(0.2)
                sys.stdout.write('\b')
    
    return
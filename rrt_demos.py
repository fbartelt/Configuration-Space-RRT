from RRT import RRT, run_RRT
import sys
sys.path.insert(1, '/home/fbartelt/Documents/UFMG/MultiRobot/Jupyterbot')
import jupyterbot as jb
import numpy as np
from collections import deque

def demo1(T=5, epochs=1000, qdes=-1, step=1e-1, P=0.8, plot_path=True, plot_rand=False, plot_next=False):
    #Meshes
    mesh_ground = jb.meshmaterial.MeshMaterial(color="#291c04",roughness=0.6, metalness=0.5, clearcoat=0, reflectivity=0.8)
    mesh_box = jb.meshmaterial.MeshMaterial(color="#302f2d",roughness=0.3, metalness=0.5, clearcoat=0.1, reflectivity=0.6)
    mesh_obs = jb.meshmaterial.MeshMaterial(color="#290410",roughness=0.8, metalness=0, clearcoat=0, reflectivity=0.8)

    #Objects
    robotA = jb.robot.Robot.create_kukakr5(name = "robotA", htm = jb.utils.Utils.trn([0,-0.01,0.3]))
    boxA = jb.box.Box(name="boxA",width=0.3,depth=0.3,height=0.3,htm=jb.utils.Utils.trn([0,-0.01,0.15]),color="white", opacity=0.6, mesh_material=mesh_box)
    boxLeft = jb.box.Box(name="boxLeft",width=0.3,depth=0.3,height=0.8,htm=jb.utils.Utils.trn([0,0.7,0.5]),color="purple", opacity=0.5, mesh_material=mesh_obs)
    boxRight = jb.box.Box(name="boxRight",width=2,depth=2,height=0.1,htm=jb.utils.Utils.trn([-0.5,0,1.6]),color="purple", opacity=0.5, mesh_material=mesh_obs)
    obstacle = jb.ball.Ball(name="obstacle",radius=0.2,htm=jb.utils.Utils.trn([0.3,-0.5,0.4]),color="purple", opacity=0.5, mesh_material=mesh_obs)
    obstacle2 = jb.box.Box(name="obstacle2",width=0.3,depth=0.3,height=1.3,htm=jb.utils.Utils.trn([0.65,0.2,0.5]),color="purple", opacity=0.5, mesh_material=mesh_obs)
    ground = jb.box.Box(name="ground", width=3,depth=3,height=0.01,htm=jb.utils.Utils.trn([0,0,0.005]),color="darkgreen", mesh_material=mesh_ground)
    obstacle3 = jb.ball.Ball(htm=jb.utils.Utils.trn([-0.2,-0.2,1]), name='obstacle3', radius=0.2, color='purple', opacity=0.5, mesh_material=mesh_obs)

    #Lights
    light1 = jb.pointlight.PointLight(name="light1", color="white", intensity=3, position=[1, 0, 1.5])
    light2 = jb.pointlight.PointLight(name="light2", color="white", intensity=3, position=[-1, 0, 1.5])
    light3 = jb.pointlight.PointLight(name="light3", color="white", intensity=3, position=[0, 0, 1.5])
    light4 = jb.pointlight.PointLight(name="light4", color="white", intensity=3, position=[0, 0.5, 0.5])

    sim = jb.simulation.Simulation()

    #qA = robotA.ikm(jb.utils.Utils.trn([0.4,5e-2, 1.4]))
    qA = np.array([ 6.43063387, -0.84815502,  4.41227638,  0.01163194,  5.86668819, 3.00612301])
    ba = jb.ball.Ball(htm=robotA.fkm(q=qA), name='init', radius=15e-3, color='#32c9bf')
    sim.add(ba)
    robotA.add_ani_frame(time=0, q=qA)

    
    sim.add(robotA)
    sim.add(boxA)
    sim.add(obstacle)
    sim.add(obstacle2)
    sim.add(obstacle3)
    sim.add(boxLeft)
    sim.add(boxRight)
    sim.add(ground)
    sim.add(light1)
    sim.add(light2)
    sim.add(light3)
    sim.add(light4)

    if qdes == -1:
        #qdes = robotA.ikm(jb.utils.Utils.trn([-0.45,-0.4,0.2]))
        qdes = np.array([3.96847265, 0.18502985, 3.75427012, 3.14593206, 7.07650027, 2.3194198])
        #qdes = np.array([3.2, 0.18502985, 3.75427012, 3.14593206, 7.07650027, 2.3194198])
        ba = jb.ball.Ball(htm=robotA.fkm(q=qdes), name='goal', radius=15e-3, color='#600d75')
    else:
        htmdes = robotA.fkm(q=qdes)
        ba = jb.ball.Ball(htm=htmdes, name='goal', radius=15e-3, color='#600d75')
    
    sim.add(ba)
    
    obstacles = [boxLeft, boxRight, obstacle, obstacle2, obstacle3]
    joint_limits = [(-155, 155), (-65, 180), (-68, 105), (-350, 350), (-130, 130), (-350, 350)]
    joint_limits = [(np.deg2rad(inf), np.deg2rad(sup)) for (inf, sup) in joint_limits]

    Tinit = RRT(robotA, qA, qdes, angles_idx=list(range(len(qA))), joint_limits=joint_limits, obstacles=obstacles, max_iter=200)
    Tgoal = RRT(robotA, qdes, qA, angles_idx=list(range(len(qA))), joint_limits=joint_limits, obstacles=obstacles, max_iter=200)
    
    trees = deque([Tinit, Tgoal])
    crand = {Tinit: '#dade10', Tgoal : '#000000'}
    cnext = {Tinit: '#071ff7', Tgoal : '#f70f07'}
    disc = True

    i=0
    while disc:
        if i%(epochs//10 or 1) == 0:
            print(i, end=' ')
        qrand = trees[0].gen_qrand(P=P)
        
        if qrand is not None:
            qnearest = trees[0].find_nearest_q(qrand)
            htmb = robotA.fkm(q=qrand)
            
            if plot_rand:
                ba = jb.ball.Ball(htm=htmb, name=f'rand_{i}', radius=1e-2, color=crand[trees[0]])
                sim.add(ba)
            
            qnext = trees[0].find_next_q(qnearest, qrand, step=step)
            
            if qnext:
                qnearest.add_children(qnext)
                if plot_next:
                    htmb = robotA.fkm(q=qnext.q)
                    ba = jb.ball.Ball(htm=htmb, name=f'ball_{i}', radius=1e-2, color=cnext[trees[0]])
                    sim.add(ba)
                
                disc = trees[0].merge(qnext, trees[1], prune=True)

        if i > epochs:
            print('Max iterations reached')
            break
        trees.rotate()
        i+=1

    if not disc:
        print('MERGED')
        dt = T / len(Tinit.path)
        #path = Tinit.pruning()[::-1]
        for i, q in enumerate(Tinit.path[::-1]):
        #for i, q in enumerate(path):
            robotA.add_ani_frame(time=i*dt, q=q)
            
            if plot_path:
                htmb = robotA.fkm(q=q)
                ba = jb.ball.Ball(htm=htmb, name=f'path_{i}', radius=1e-2, color='#d15706')
                sim.add(ba)
        sim.run()

    return sim, (robotA, obstacles, ground, boxA, Tinit, Tgoal)

def demo2(T=5, epochs=1000, qdes=-1, step=1e-1, P=0.8, plot_path=True, plot_rand=False, plot_next=False):
    #Meshes
    mesh_ground = jb.meshmaterial.MeshMaterial(color="#291c04",roughness=0.6, metalness=0.5, clearcoat=0, reflectivity=0.8)
    mesh_box = jb.meshmaterial.MeshMaterial(color="#302f2d",roughness=0.3, metalness=0.5, clearcoat=0.1, reflectivity=0.6)
    mesh_obs = jb.meshmaterial.MeshMaterial(color="#290410",roughness=0.8, metalness=0, clearcoat=0, reflectivity=0.8)

    #Objects
    robotA = jb.robot.Robot.create_kukakr5(name = "robotA", htm = jb.utils.Utils.trn([0,-0.01,0.3]))
    boxA = jb.box.Box(name="boxA",width=0.3,depth=0.3,height=0.3,htm=jb.utils.Utils.trn([0,-0.01,0.15]),color="white", opacity=0.6, mesh_material=mesh_box)
    obstacle = jb.ball.Ball(name="obstacle",radius=0.2,htm=jb.utils.Utils.trn([0.3,-0.5,0.2]),color="green", opacity=0.5, mesh_material=mesh_obs)
    obstacle2 = jb.ball.Ball(name="obstacle2",radius=0.2,htm=jb.utils.Utils.trn([-0.4,-0.7,0.9]),color="cyan", opacity=0.5, mesh_material=mesh_obs)
    ground = jb.box.Box(name="ground", width=3,depth=3,height=0.01,htm=jb.utils.Utils.trn([0,0,0.005]), mesh_material=mesh_ground, color="#4a3107")
    obstacle3 = jb.ball.Ball(htm=jb.utils.Utils.trn([0.9,0,0.4]), name='obstacle3', radius=0.2, color='red', opacity=0.5, mesh_material=mesh_obs)
    obstacle4 = jb.ball.Ball(name="obstacle7",radius=0.14,htm=jb.utils.Utils.trn([0.3,-0.5,1.2]),color="purple", opacity=0.5, mesh_material=mesh_obs)
    obstacle5 = jb.ball.Ball(name="obstacle5",radius=0.2,htm=jb.utils.Utils.trn([-0.2,-0.4, 0.1]),color="black", opacity=0.5, mesh_material=mesh_obs)

    #qA = np.array([ -5, -0.84815502,  5.4,  0.01163194,  5.86668819, 3.00612301])
    qA = np.array([-2.7, -0.5, -1.187, -6.109, 0.78, -6.109])
    ba = jb.ball.Ball(htm=robotA.fkm(q=qA), name='init', radius=15e-3, color='#32c9bf')
    #sim.add(robotA)
    #sim.add(ba)
    #sim.add(boxA)
    #sim.add(obstacle)
    #sim.add(obstacle2)
    #sim.add(obstacle3)
    #sim.add(obstacle4)
    #sim.add(obstacle5)
    #sim.add(ground)

    robotA.add_ani_frame(time=0, q=qA)
  
    light1 = jb.pointlight.PointLight(name="light1", color="white", intensity=3, position=[1, 0, 1.5])
    light2 = jb.pointlight.PointLight(name="light2", color="white", intensity=3, position=[-1, 0, 1.5])
    light3 = jb.pointlight.PointLight(name="light3", color="white", intensity=3, position=[0, 0, 1.5])
    light4 = jb.pointlight.PointLight(name="light4", color="white", intensity=3, position=[0, 0.5, 0.5])
    #sim.add(light1)
    #sim.add(light2)
    #sim.add(light3)
    #sim.add(light4)
    sim = jb.simulation.Simulation([robotA, ba, boxA, ground, light1, light2, light3, light4, obstacle, obstacle2, obstacle3, obstacle4, obstacle5])

    if qdes == -1:
        #qdes = robotA.ikm(jb.utils.Utils.trn([-0.45,-0.4,0.2]))
        #qdes = np.array([3.2, 0.18502985, 3.8, 1.15, 7.07650027, 2.3194198])
        qdes = np.array([0.52, 0.22, -0.94, 6.10, 0, 6.1])
        #qdes = np.array([3.2, 0.18502985, 3.75427012, 3.14593206, 7.07650027, 2.3194198])
        ba = jb.ball.Ball(htm=robotA.fkm(q=qdes), name='goal', radius=15e-3, color='#600d75')
    else:
        htmdes = robotA.fkm(q=qdes)
        ba = jb.ball.Ball(htm=htmdes, name='goal', radius=15e-3, color='#600d75')
    
    sim.add(ba)
    
    obstacles = [obstacle, obstacle2, obstacle3, obstacle5]
    joint_limits = [(-155, 155), (-65, 180), (-68, 105), (-350, 350), (-130, 130), (-350, 350)]
    #joint_limits = [(-350, 350), (-65, 180), (-68, 105), (-350, 350), (-130, 130), (-350, 350)]
    joint_limits = [(np.deg2rad(inf), np.deg2rad(sup)) for (inf, sup) in joint_limits]

    Tinit = RRT(robotA, qA, qdes, angles_idx=list(range(len(qA))), joint_limits=joint_limits, obstacles=obstacles, max_iter=200)
    Tgoal = RRT(robotA, qdes, qA, angles_idx=list(range(len(qA))), joint_limits=joint_limits, obstacles=obstacles, max_iter=200)
    
    trees = deque([Tinit, Tgoal])
    crand = {Tinit: '#dade10', Tgoal : '#000000'}
    cnext = {Tinit: '#071ff7', Tgoal : '#f70f07'}
    disc = True

    i=0
    while disc:
        if i%(epochs//10 or 1) == 0:
            print(i, end=' ')
        qrand = trees[0].gen_qrand(P=P)
        
        if qrand is not None:
            qnearest = trees[0].find_nearest_q(qrand)
            htmb = robotA.fkm(q=qrand)
            
            if plot_rand:
                ba = jb.ball.Ball(htm=htmb, name=f'rand_{i}', radius=1e-2, color=crand[trees[0]])
                sim.add(ba)
            
            qnext = trees[0].find_next_q(qnearest, qrand, step=step)
            
            if qnext:
                qnearest.add_children(qnext)
                if plot_next:
                    htmb = robotA.fkm(q=qnext.q)
                    ba = jb.ball.Ball(htm=htmb, name=f'ball_{i}', radius=1e-2, color=cnext[trees[0]])
                    sim.add(ba)
                
                disc = trees[0].merge(qnext, trees[1], prune=True)

        if i > epochs:
            print('Max iterations reached')
            break
        trees.rotate()
        i+=1

    if not disc:
        print('MERGED')
        dt = T / len(Tinit.path)
        #path = Tinit.pruning()[::-1]
        for i, q in enumerate(Tinit.path[::-1]):
        #for i, q in enumerate(path):
            robotA.add_ani_frame(time=i*dt, q=q)
            
            if plot_path:
                htmb = robotA.fkm(q=q)
                ba = jb.ball.Ball(htm=htmb, name=f'path_{i}', radius=1e-2, color='#d15706')
                sim.add(ba)
        sim.run()

    return sim, (robotA, obstacles, ground, Tinit, Tgoal)
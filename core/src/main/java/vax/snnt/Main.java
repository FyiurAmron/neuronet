package vax.snnt;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import vax.snnt.neuronet.*;
import static vax.snnt.neuronet.Rng.rnd;
import static vax.snnt.neuronet.TransferFunction.*;

/**

 @author toor
 */
public class Main {

    /**
     @param args the command line arguments
     */
    public static void main ( String[] args ) {
        test6();
    }

    // 0-layer network
    public static void test1 () {
        Neuron inN = new Neuron(
                new TransferFunction( 0.5, 0.5, 0, 0, FunctionType.Lin ),
                new TransferFunction( 2, 2, 0, 0, FunctionType.Lin ) );
        Neuron outN = new Neuron(
                new TransferFunction( 0.25, 2, 0, 0, FunctionType.Lin ),
                new TransferFunction( 0.5, 4, 0, 0, FunctionType.Lin ),
                System.out::println );
        inN.addNeuronOutput( outN, 2.0 );

        inN.add( 0.42 );
        inN.process();
        outN.process(); // 0.84
    }

    // 1-layer network
    public static void test2 () {
        Neuron inN = new Neuron(
                new TransferFunction( 0.5, 0.5, 0, 0, FunctionType.Lin ),
                new TransferFunction( 2, 2, 0, 0, FunctionType.Lin ) );
        Neuron outN = new Neuron(
                new TransferFunction( 0.25, 2, 0, 0, FunctionType.Lin ),
                new TransferFunction( 0.5, 4, 0, 0, FunctionType.Lin ),
                System.out::println );

        NetLayer midLayer = new NetLayer();
        for( int i = 0; i < 10; i++ ) {
            Neuron neuron = new Neuron( IDENTITY, IDENTITY );
            midLayer.addNeuron( neuron );
        }

        NetLayer inputLayer = new NetLayer();
        inputLayer.addNeuron( inN );

        NetLayer outputLayer = new NetLayer();
        outputLayer.addNeuron( outN );

        for( Neuron neuron : midLayer.getNeurons() ) {
            inN.addNeuronOutput( neuron, 1.0 );
            neuron.addNeuronOutput( outN, 0.05 );
        }

        Network network = new Network();
        network.addLayer( inputLayer );
        network.addLayer( midLayer );
        network.addLayer( outputLayer );

        inN.add( 0.42 );
        network.process();
        // v = 0.20999999999999996
    }

    // 2-layer network
    public static void test3 () {
        Neuron inN = new Neuron(
                new TransferFunction( 0.5, 0.5, 0, 0, FunctionType.Lin ),
                new TransferFunction( 2, 2, 0, 0, FunctionType.Lin ) );
        Neuron outN = new Neuron(
                new TransferFunction( 0.25, 2, 0, 0, FunctionType.Lin ),
                new TransferFunction( 0.5, 4, 0, 0, FunctionType.Lin ),
                System.out::println );

        NetLayer midLayer1 = new NetLayer();
        for( int i = 0; i < 10; i++ ) {
            Neuron neuron = new Neuron( IDENTITY, IDENTITY );
            midLayer1.addNeuron( neuron );
        }

        NetLayer midLayer2 = new NetLayer();
        for( int i = 0; i < 10; i++ ) {
            Neuron neuron = new Neuron( IDENTITY, IDENTITY );
            midLayer2.addNeuron( neuron );
        }

        NetLayer inputLayer = new NetLayer();
        inputLayer.addNeuron( inN );

        NetLayer outputLayer = new NetLayer();
        outputLayer.addNeuron( outN );

        for( Neuron neuron : midLayer1.getNeurons() ) {
            inN.addNeuronOutput( neuron, 1.0 );
            for( Neuron neuron2 : midLayer2.getNeurons() ) {
                neuron.addNeuronOutput( neuron2, 0.1 );
            }
        }

        for( Neuron neuron : midLayer2.getNeurons() ) {
            neuron.addNeuronOutput( outN, 0.05 );
        }

        Network network = new Network();
        network.addLayer( inputLayer );
        network.addLayer( midLayer1 );
        network.addLayer( midLayer2 );
        network.addLayer( outputLayer );

        inN.add( 0.42 );
        network.process();
        // v = 0.20999999999999996
    }

    private static Network createTestTwoLayerNetwork1 () {
        Network network = new Network();

        Neuron inN = new Neuron( IDENTITY, IDENTITY );
        Neuron outN = new Neuron( IDENTITY, IDENTITY,
                network::collectOutput );

        int neuronsInLayer = 10;

        NetLayer midLayer1 = new NetLayer();
        for( int i = 0; i < neuronsInLayer; i++ ) {
            Neuron neuron = new Neuron(
                    new TransferFunction( rnd(), rnd(), 0, 0, FunctionType.Lin ),
                    new TransferFunction( rnd(), rnd(), 0, 0, FunctionType.Lin ) );
            midLayer1.addNeuron( neuron );
        }

        NetLayer midLayer2 = new NetLayer();
        for( int i = 0; i < neuronsInLayer; i++ ) {
            Neuron neuron = new Neuron(
                    new TransferFunction( rnd(), rnd(), 0, 0, FunctionType.Lin ),
                    new TransferFunction( rnd(), rnd(), 0, 0, FunctionType.Lin ) );
            midLayer2.addNeuron( neuron );
        }

        NetLayer inputLayer = new NetLayer();
        inputLayer.addNeuron( inN );

        NetLayer outputLayer = new NetLayer();
        outputLayer.addNeuron( outN );

        for( Neuron neuron : midLayer1.getNeurons() ) {
            inN.addNeuronOutput( neuron, 1.0 );
            for( Neuron neuron2 : midLayer2.getNeurons() ) {
                neuron.addNeuronOutput( neuron2, 2.0 / neuronsInLayer * Rng.nextDouble() );
            }
        }

        for( Neuron neuron : midLayer2.getNeurons() ) {
            neuron.addNeuronOutput( outN, 2.0 / neuronsInLayer * Rng.nextDouble() );
        }

        network.addLayer( inputLayer );
        network.addLayer( midLayer1 );
        network.addLayer( midLayer2 );
        network.addLayer( outputLayer );

        return network;
    }

    // 2-layer network population (10k)
    public static void test4 () {
        double[][] inputs = {
            { -1.0 }, { -0.9 }, { -0.8 }, { -0.7 }, { -0.6 }, { -0.5 }, { -0.4 }, { -0.3 }, { -0.2 }, { -0.1 },
            { 0 },
            { 0.1 }, { 0.2 }, { 0.3 }, { 0.4 }, { 0.5 }, { 0.6 }, { 0.7 }, { 0.8 }, { 0.9 }, { 1.0 }
        };
        double[][] expectedOutputs = {
            { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 },
            { 0 },
            { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }
        };
        int inputCount = inputs.length;
        int popCount = 10_000, bestCount = 100;

        for( Rng.rndMax = 2.0; Rng.rndMax < 4.0; Rng.rndMax += 0.1 ) {
            System.out.println( "Rng.rndMax = " + Rng.rndMax );
            ArrayList<Network> population = new ArrayList<>();
            for( int i = 0; i < popCount; i++ ) {
                population.add( createTestTwoLayerNetwork1() );
            }
            int i = 0;
            for( Network network : population ) {
                for( int j = 0; j < inputCount; j++ ) {
                    network.addInputs( inputs[j] );
                    network.process();
                    network.calcOutputDist( expectedOutputs[j] );
                    /*
                     for( double d : network.getOutputs() ) {
                     System.out.println( d );
                     }
                     */
                }
                //System.out.println( "pop " + i + " distL1 = " + distL1 + " distL2 = " + distL2 );
                i++;
            }
            population.sort( (Network o1, Network o2)
                    -> ( o1.getDistL2() < o2.getDistL2() ) ? -1 : ( ( o1.getDistL2() > o2.getDistL2() ) ? 1 : 0 ) );

            double sumL2 = 0;
            i = 0;
            for( Network network : population ) {
                //System.out.println( "pop " + i + " distL1 = " + network.getDistL1() + " distL2 = " + network.getDistL2() );
                sumL2 += network.getDistL2();
                i++;
            }
            System.out.println( "L2avg = " + sumL2 / popCount );

            population = new ArrayList<>( population.subList( 0, bestCount ) );

            sumL2 = 0;
            i = 0;
            for( Network network : population ) {
                //System.out.println( "pop " + i + " distL1 = " + network.getDistL1() + " distL2 = " + network.getDistL2() );
                sumL2 += network.getDistL2();
                i++;
            }
            System.out.println( "L2avg_best = " + sumL2 / bestCount );
            //mix/mutate population
        }
        // expected best @ Rng.rndMax = 2.5
    }

    /*
     private static TransferFunction createRandomTransferFunction () {
     switch ( rng.nextInt( 3 ) ) {
     case 1:
     return new TransferFunction.Log(rnd(), rnd() );
     case 2:
     return new TransferFunction.Log( rnd(), rnd() );
     //return new TransferFunction.Exp( rnd(), rnd() );
     case 0:
     return new TransferFunction.Log( rnd(), rnd() );
     //return new TransferFunction.Lin( rnd(), rnd() );

     }
     throw new RuntimeException();
     }
     */
    private static TransferFunction createRandomTransferFunction () {
        switch ( Rng.nextInt( 3 ) ) {
            case 1:
                return new TransferFunction( rnd(), rnd(), rnd(), rnd(), FunctionType.Lin );
            case 2:
                return new TransferFunction( rnd(), rnd(), rnd(), rnd(), FunctionType.Log );
            //return new TransferFunction( rnd(), rnd(), 0, 0, FunctionType.Log );
            case 0:
                return new TransferFunction( rnd(), rnd(), rnd(), rnd(), FunctionType.Log );
            //return new TransferFunction( rnd(), rnd(), 0, 0, FunctionType.Exp );

        }
        throw new RuntimeException();
    }

    private static Network createTestTwoLayerNetwork2 () {
        Network network = new Network();

        Neuron inN = new Neuron( IDENTITY, IDENTITY );
        Neuron outN = new Neuron( IDENTITY, IDENTITY,
                network::collectOutput );

        int neuronsInLayer1 = 5, neuronsInLayer2 = 5;

        NetLayer midLayer1 = new NetLayer();
        for( int i = 0; i < neuronsInLayer1; i++ ) {
            Neuron neuron = new Neuron( createRandomTransferFunction(), createRandomTransferFunction() );
            midLayer1.addNeuron( neuron );
        }

        NetLayer midLayer2 = new NetLayer();
        for( int i = 0; i < neuronsInLayer2; i++ ) {
            Neuron neuron = new Neuron( createRandomTransferFunction(), createRandomTransferFunction() );
            midLayer2.addNeuron( neuron );
        }

        NetLayer inputLayer = new NetLayer();
        inputLayer.addNeuron( inN );

        NetLayer outputLayer = new NetLayer();
        outputLayer.addNeuron( outN );

        for( Neuron neuron : midLayer1.getNeurons() ) {
            inN.addNeuronOutput( neuron, 1.0 );
            for( Neuron neuron2 : midLayer2.getNeurons() ) {
                //neuron.addNeuronOutput( neuron2, 2.0 / neuronsInLayer1 * Rng.nextDouble()  );
                neuron.addNeuronOutput( neuron2, 1.0 / neuronsInLayer1 );
            }
        }

        for( Neuron neuron : midLayer2.getNeurons() ) {
            //neuron.addNeuronOutput( outN, 2.0 / neuronsInLayer2 * Rng.nextDouble()  );
            neuron.addNeuronOutput( outN, 1.0 / neuronsInLayer2 );
        }

        network.addLayer( inputLayer );
        network.addLayer( midLayer1 );
        network.addLayer( midLayer2 );
        network.addLayer( outputLayer );

        return network;
    }

    private static double[][] valuesLinear = {
        { -1.0 }, { -0.9 }, { -0.8 }, { -0.7 }, { -0.6 }, { -0.5 }, { -0.4 }, { -0.3 }, { -0.2 }, { -0.1 },
        { 0 },
        { 0.1 }, { 0.2 }, { 0.3 }, { 0.4 }, { 0.5 }, { 0.6 }, { 0.7 }, { 0.8 }, { 0.9 }, { 1.0 }
    };
    private static double[][] valuesStep = {
        { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 },
        { 0 },
        { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }
    };

    public static void test5 () {
        double[][] inputs = valuesLinear;
        double[][] expectedOutputs = valuesStep;

        int inputCount = inputs.length;
        int popCount = 100_000, bestCount = 100;

        for( Rng.rndMax = 3.0; Rng.rndMax < 6.0; Rng.rndMax += 0.1 ) {
            System.out.println( "Rng.rndMax = " + Rng.rndMax );
            ArrayList<Network> population = new ArrayList<>();
            for( int i = 0; i < popCount; i++ ) {
                population.add( createTestTwoLayerNetwork2() );
            }
            int i = 0;
            for( Network network : population ) {
                for( int j = 0; j < inputCount; j++ ) {
                    network.addInputs( inputs[j] );
                    network.process();
                    network.calcOutputDist( expectedOutputs[j] );
                    /*
                     for( double d : network.getOutputs() ) {
                     System.out.println( d );
                     }
                     */
                }
                //System.out.println( "pop " + i + " distL1 = " + distL1 + " distL2 = " + distL2 );
                i++;
            }
            population.sort( (Network o1, Network o2)
                    -> ( o1.getDistL2() < o2.getDistL2() ) ? -1 : ( ( o1.getDistL2() > o2.getDistL2() ) ? 1 : 0 ) );

            double sumL2 = 0;
            i = 0;
            for( Network network : population ) {
                //System.out.println( "pop " + i + " distL1 = " + network.getDistL1() + " distL2 = " + network.getDistL2() );
                sumL2 += network.getDistL2();
                i++;
            }
            System.out.println( "L2avg = " + sumL2 / popCount );

            population = new ArrayList<>( population.subList( 0, bestCount ) );

            sumL2 = 0;
            i = 0;
            for( Network network : population ) {
                //System.out.println( "pop " + i + " distL1 = " + network.getDistL1() + " distL2 = " + network.getDistL2() );
                sumL2 += network.getDistL2();
                i++;
            }
            System.out.println( "L2avg_best = " + sumL2 / bestCount );
            //mix/mutate population
        }
    }

    public static void test6 () {
        GsonBuilder gb = new GsonBuilder();
        gb.setPrettyPrinting();
        Gson gson = gb.create();

        double[][] inputs = valuesLinear;
        double[][] expectedOutputs = valuesLinear;

        int inputCount = inputs.length;
        int popCount = 10_000, bestCount = 100;

        Rng.rndMax = 5.0;
        System.out.println( "Rng.rndMax = " + Rng.rndMax );
        ArrayList<Network> population = new ArrayList<>();
        for( int i = 0; i < popCount; i++ ) {
            population.add( createTestTwoLayerNetwork2() );
        }
        int i = 0;
        for( Network network : population ) {
            for( int j = 0; j < inputCount; j++ ) {
                network.addInputs( inputs[j] );
                network.process();
                network.calcOutputDist( expectedOutputs[j] );
                /*
                 for( double d : network.getOutputs() ) {
                 System.out.println( d );
                 }
                 */
            }
            //System.out.println( "pop " + i + " distL1 = " + distL1 + " distL2 = " + distL2 );
            i++;
        }
        population.sort( (Network o1, Network o2)
                -> ( o1.getDistL2() < o2.getDistL2() ) ? -1 : ( ( o1.getDistL2() > o2.getDistL2() ) ? 1 : 0 ) );

        double sumL2 = 0;
        i = 0;
        for( Network network : population ) {
            //System.out.println( "pop " + i + " distL1 = " + network.getDistL1() + " distL2 = " + network.getDistL2() );
            sumL2 += network.getDistL2();
            i++;
        }
        System.out.println( "L2avg = " + sumL2 / popCount );

        Network worstNetwork = population.get( population.size() - 1 );

        try ( PrintWriter out = new PrintWriter( "output/worstANN.json" ); ) {
            out.print( gson.toJson( worstNetwork ) );
        } catch (FileNotFoundException ex) {
            throw new RuntimeException( ex );
        }

        ArrayList<Network> newPopulation = new ArrayList<>();
        int mutations = popCount / bestCount;
        for( Network network : population.subList( 0, bestCount ) ) {
            newPopulation.add( network );
            for( int j = 0; j < mutations; j++ ) {
                newPopulation.add( network.mutate( Rng.rng, 0.1, 0.1, 0.1 ) );
            }
        }
        population = newPopulation;
        // TODO mix/mutate population properly

        sumL2 = 0;
        i = 0;
        for( Network network : population ) {
            System.out.println( "pop " + i + " distL1 = " + network.getDistL1() + " distL2 = " + network.getDistL2() );
            sumL2 += network.getDistL2();
            i++;
        }
        System.out.println( "L2avg_best = " + sumL2 / bestCount );

        Network bestNetwork = population.get( 0 );

        try ( PrintWriter out = new PrintWriter( "output/bestANN.json" ); ) {
            out.print( gson.toJson( bestNetwork ) );
        } catch (FileNotFoundException ex) {
            throw new RuntimeException( ex );
        }

        // TODO save weighs
        double[] inputArr = { 0 };
        try ( PrintWriter out = new PrintWriter( "output/wideBestResults.txt" ); ) {
            out.println( "IN,OUT" );
            for( double d = -10; d < 10; d = Math.round( ( d + 0.01 ) * 100 ) / 100.0 ) {
                inputArr[0] = d;
                bestNetwork.addInputs( inputArr );
                bestNetwork.process();
                double output = bestNetwork.getOutputs()[0];
                out.println( d + "," + output );
            }
        } catch (FileNotFoundException ex) {
            throw new RuntimeException( ex );
        }

        try ( PrintWriter out = new PrintWriter( "output/wideWorstResults.txt" ); ) {
            out.println( "IN,OUT" );
            for( double d = -10; d < 10; d += 0.01 ) {
                inputArr[0] = d;
                worstNetwork.addInputs( inputArr );
                worstNetwork.process();
                double output = worstNetwork.getOutputs()[0];
                out.println( d + "," + output );
            }
        } catch (FileNotFoundException ex) {
            throw new RuntimeException( ex );
        }
    }

}

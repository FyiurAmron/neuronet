package vax.snnt;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import vax.snnt.neuronet.*;
import vax.snnt.neuronet.Neuron.*;

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
        MidNeuron inN = new Neuron.MidNeuron( new TransferFunction.Lin( 0.5, 0.5 ), new TransferFunction.Lin( 2, 2 ) );
        OutputNeuron outN = new Neuron.OutputNeuron( new TransferFunction.Lin( 0.25, 2 ), new TransferFunction.Lin( 0.5, 4 ),
                System.out::println );
        inN.addNeuronOutput( outN, 2.0 );

        inN.add( 0.42 );
        inN.process();
        outN.process(); // 0.84
    }

    // 1-layer network
    public static void test2 () {
        MidNeuron inN = new Neuron.MidNeuron( new TransferFunction.Lin( 0.5, 0.5 ), new TransferFunction.Lin( 2, 2 ) );
        OutputNeuron outN = new Neuron.OutputNeuron( new TransferFunction.Lin( 0.25, 2 ), new TransferFunction.Lin( 0.5, 4 ),
                System.out::println );

        Layer<MidNeuron> midLayer = new Layer<>();
        TransferFunction identity = new TransferFunction.Lin( 1, 1 );
        for( int i = 0; i < 10; i++ ) {
            MidNeuron neuron = new MidNeuron( identity, identity );
            midLayer.addNeuron( neuron );
        }

        Layer<MidNeuron> inputLayer = new Layer<>();
        inputLayer.addNeuron( inN );

        Layer<OutputNeuron> outputLayer = new Layer<>();
        outputLayer.addNeuron( outN );

        for( MidNeuron neuron : midLayer.getNeurons() ) {
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
        MidNeuron inN = new Neuron.MidNeuron( new TransferFunction.Lin( 0.5, 0.5 ), new TransferFunction.Lin( 2, 2 ) );
        OutputNeuron outN = new Neuron.OutputNeuron( new TransferFunction.Lin( 0.25, 2 ), new TransferFunction.Lin( 0.5, 4 ),
                System.out::println );

        TransferFunction identity = new TransferFunction.Lin( 1, 1 );

        Layer<MidNeuron> midLayer1 = new Layer<>();
        for( int i = 0; i < 10; i++ ) {
            MidNeuron neuron = new MidNeuron( identity, identity );
            midLayer1.addNeuron( neuron );
        }

        Layer<MidNeuron> midLayer2 = new Layer<>();
        for( int i = 0; i < 10; i++ ) {
            MidNeuron neuron = new MidNeuron( identity, identity );
            midLayer2.addNeuron( neuron );
        }

        Layer<MidNeuron> inputLayer = new Layer<>();
        inputLayer.addNeuron( inN );

        Layer<OutputNeuron> outputLayer = new Layer<>();
        outputLayer.addNeuron( outN );

        for( MidNeuron neuron : midLayer1.getNeurons() ) {
            inN.addNeuronOutput( neuron, 1.0 );
            for( MidNeuron neuron2 : midLayer2.getNeurons() ) {
                neuron.addNeuronOutput( neuron2, 0.1 );
            }
        }

        for( MidNeuron neuron : midLayer2.getNeurons() ) {
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

    private static final Random rng = new Random( 1410 );
    private static double rndMax = 3.5; // 2.5 // 3.5

    private static double rnd () {
        return ( rng.nextFloat() - 0.5 ) * 2 * rndMax;
    }

    private static Network createTestTwoLayerNetwork1 () {
        Network network = new Network();

        TransferFunction identity = new TransferFunction.Lin( 1, 1 );

        MidNeuron inN = new Neuron.MidNeuron( identity, identity );
        OutputNeuron outN = new Neuron.OutputNeuron( identity, identity,
                network::collectOutput );

        int neuronsInLayer = 10;

        Layer<MidNeuron> midLayer1 = new Layer<>();
        for( int i = 0; i < neuronsInLayer; i++ ) {
            MidNeuron neuron = new MidNeuron( new TransferFunction.Lin( rnd(), rnd() ), new TransferFunction.Lin( rnd(), rnd() ) );
            midLayer1.addNeuron( neuron );
        }

        Layer<MidNeuron> midLayer2 = new Layer<>();
        for( int i = 0; i < neuronsInLayer; i++ ) {
            MidNeuron neuron = new MidNeuron( new TransferFunction.Lin( rnd(), rnd() ), new TransferFunction.Lin( rnd(), rnd() ) );
            midLayer2.addNeuron( neuron );
        }

        Layer<MidNeuron> inputLayer = new Layer<>();
        inputLayer.addNeuron( inN );

        Layer<OutputNeuron> outputLayer = new Layer<>();
        outputLayer.addNeuron( outN );

        for( MidNeuron neuron : midLayer1.getNeurons() ) {
            inN.addNeuronOutput( neuron, 1.0 );
            for( MidNeuron neuron2 : midLayer2.getNeurons() ) {
                neuron.addNeuronOutput( neuron2, 2.0 / neuronsInLayer * rng.nextFloat() );
            }
        }

        for( MidNeuron neuron : midLayer2.getNeurons() ) {
            neuron.addNeuronOutput( outN, 2.0 / neuronsInLayer * rng.nextFloat() );
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

        for( rndMax = 2.0; rndMax < 4.0; rndMax += 0.1 ) {
            System.out.println( "rndMax = " + rndMax );
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
        // expected best @ rndMax = 2.5
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
        switch ( rng.nextInt( 3 ) ) {
            case 1:
                return new TransferFunction( rnd(), rnd(), new TransferFunction.LinOp() );
            case 2:
                return new TransferFunction( rnd(), rnd(), new TransferFunction.LinOp() );
            //return new TransferFunction.Exp( rnd(), rnd() );
            case 0:
                return new TransferFunction( rnd(), rnd(), new TransferFunction.LinOp() );
            //return new TransferFunction.Lin( rnd(), rnd() );

        }
        throw new RuntimeException();
    }

    private static Network createTestTwoLayerNetwork2 () {
        Network network = new Network();

        TransferFunction identity = new TransferFunction( 1, 1, new TransferFunction.LinOp() );

        MidNeuron inN = new Neuron.MidNeuron( identity, identity );
        OutputNeuron outN = new Neuron.OutputNeuron( identity, identity,
                network::collectOutput );

        int neuronsInLayer1 = 10, neuronsInLayer2 = 10;

        Layer<MidNeuron> midLayer1 = new Layer<>();
        for( int i = 0; i < neuronsInLayer1; i++ ) {
            MidNeuron neuron = new MidNeuron( createRandomTransferFunction(), createRandomTransferFunction() );
            midLayer1.addNeuron( neuron );
        }

        Layer<MidNeuron> midLayer2 = new Layer<>();
        for( int i = 0; i < neuronsInLayer2; i++ ) {
            MidNeuron neuron = new MidNeuron( createRandomTransferFunction(), createRandomTransferFunction() );
            midLayer2.addNeuron( neuron );
        }

        Layer<MidNeuron> inputLayer = new Layer<>();
        inputLayer.addNeuron( inN );

        Layer<OutputNeuron> outputLayer = new Layer<>();
        outputLayer.addNeuron( outN );

        for( MidNeuron neuron : midLayer1.getNeurons() ) {
            inN.addNeuronOutput( neuron, 1.0 );
            for( MidNeuron neuron2 : midLayer2.getNeurons() ) {
                //neuron.addNeuronOutput( neuron2, 2.0 / neuronsInLayer1 * rng.nextFloat() );
                neuron.addNeuronOutput( neuron2, 1.0 / neuronsInLayer1 );
            }
        }

        for( MidNeuron neuron : midLayer2.getNeurons() ) {
            //neuron.addNeuronOutput( outN, 2.0 / neuronsInLayer2 * rng.nextFloat() );
            neuron.addNeuronOutput( outN, 1.0 / neuronsInLayer2 );
        }

        network.addLayer( inputLayer );
        network.addLayer( midLayer1 );
        network.addLayer( midLayer2 );
        network.addLayer( outputLayer );

        return network;
    }

    public static void test5 () {
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

        for( rndMax = 3.0; rndMax < 6.0; rndMax += 0.1 ) {
            System.out.println( "rndMax = " + rndMax );
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

        double[][] inputs = {
            { -1.0 }, { -0.9 }, { -0.8 }, { -0.7 }, { -0.6 }, { -0.5 }, { -0.4 }, { -0.3 }, { -0.2 }, { -0.1 },
            { 0 },
            { 0.1 }, { 0.2 }, { 0.3 }, { 0.4 }, { 0.5 }, { 0.6 }, { 0.7 }, { 0.8 }, { 0.9 }, { 1.0 }
        };
        double[][] expectedOutputs = {
            { -1.0 }, { -3.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 }, { -1.0 },
            { 0 },
            { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }, { 1.0 }
        };
        int inputCount = inputs.length;
        int popCount = 10_000, bestCount = 100;

        rndMax = 5.0;
        System.out.println( "rndMax = " + rndMax );
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

        try ( PrintWriter out = new PrintWriter( "worstANN.json" ); ) {
            out.print( gson.toJson( worstNetwork ) );
        } catch (FileNotFoundException ex) {
            throw new RuntimeException( ex );
        }

        population = new ArrayList<>( population.subList( 0, bestCount ) );
        // TODO mix/mutate population

        sumL2 = 0;
        i = 0;
        for( Network network : population ) {
            System.out.println( "pop " + i + " distL1 = " + network.getDistL1() + " distL2 = " + network.getDistL2() );
            sumL2 += network.getDistL2();
            i++;
        }
        System.out.println( "L2avg_best = " + sumL2 / bestCount );

        Network bestNetwork = population.get( 0 );

        try ( PrintWriter out = new PrintWriter( "bestANN.json" ); ) {
            out.print( gson.toJson( bestNetwork ) );
        } catch (FileNotFoundException ex) {
            throw new RuntimeException( ex );
        }

        // TODO save weighs
        double[] inputArr = { 0 };
        try ( PrintWriter out = new PrintWriter( "wideBestResults.txt" ); ) {
            out.println( "IN,OUT" );
            for( double d = -10; d < 10; d += 0.01 ) {
                inputArr[0] = d;
                bestNetwork.addInputs( inputArr );
                bestNetwork.process();
                double output = bestNetwork.getOutputs()[0];
                out.println( d + "," + output );
            }
        } catch (FileNotFoundException ex) {
            throw new RuntimeException( ex );
        }

        try ( PrintWriter out = new PrintWriter( "wideWorstResults.txt" ); ) {
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

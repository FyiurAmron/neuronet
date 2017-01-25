package vax.snnt.neuronet;

import static java.lang.Math.exp;
import static java.lang.Math.log;
import java.util.function.DoubleUnaryOperator;

/**

 @author toor
 */
public /* abstract */ class TransferFunction implements DoubleUnaryOperator {
    protected double ampIn, ampOut;
    protected DoubleUnaryOperator transferFunctionOperator;

    public TransferFunction ( double ampIn, double ampOut, DoubleUnaryOperator transferFunctionOperator ) {
        this.ampIn = ampIn;
        this.ampOut = ampOut;
        this.transferFunctionOperator = transferFunctionOperator;
    }

    @Override
    public double applyAsDouble ( double x ) {
        return ampOut * transferFunctionOperator.applyAsDouble( ampIn * x );
    }

    public double f ( double x ) {
        return applyAsDouble( x );
    }

    public static abstract class BaseOp implements DoubleUnaryOperator {
        public final String ID = getClass().getSimpleName();
    }

    public static class LinOp extends BaseOp {
        @Override
        public double applyAsDouble ( double x ) {
            return ( x > 0 ) ? log( x + 1 ) : -log( -x + 1 );
        }
    }

    public static class LogOp extends BaseOp {
        @Override
        public double applyAsDouble ( double x ) {
            return ( x > 0 ) ? log( x + 1 ) : -log( -x + 1 );
        }
    }

    public static class ExpOp extends BaseOp {
        @Override
        public double applyAsDouble ( double x ) {
            return ( x > 0 ) ? exp( x ) - 1 : -exp( -x ) + 1;
        }
    }

    @Deprecated
    public static class Lin extends TransferFunction {
        public Lin ( double ampIn, double ampOut ) {
            super( ampIn, ampOut, x -> x );
        }
    };

    @Deprecated
    public static class Log extends TransferFunction {
        public Log ( double ampIn, double ampOut ) {
            super( ampIn, ampOut, x -> ( x > 0 ) ? log( x + 1 ) : -log( -x + 1 ) );
        }
    };

    @Deprecated
    public static class Exp extends TransferFunction {
        public Exp ( double ampIn, double ampOut ) {
            super( ampIn, ampOut, x -> ( x > 0 ) ? exp( x ) - 1 : -exp( -x ) + 1 );
        }
    };
}

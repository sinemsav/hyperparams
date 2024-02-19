package comparator

import (
	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/dckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
	"github.com/tuneinsight/lattigo/v4/utils"
)

type DebugToolbox struct {
	Encryptor            rlwe.Encryptor
	Decryptor            rlwe.Decryptor // only used for dummy bootstrapping
	Encoder              ckks.Encoder
	Evaluator            ckks.Evaluator
	CRS                  *utils.KeyedPRNG
	BootstrappingParties []*Party
	MinLevel             int
	LogBound             uint
}

// Party bootstrapping parameters
type Party struct {
	*dckks.RefreshProtocol
	SK    *rlwe.SecretKey
	Share *dckks.RefreshShare
}
